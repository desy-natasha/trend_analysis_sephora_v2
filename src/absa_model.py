import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from transformers import logging as hf_logging
from tqdm import tqdm
from unittest.mock import patch

# Suppress warnings
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
warnings.filterwarnings("ignore",message="Precision is ill-defined",category=UserWarning)
tqdm.__init__ = lambda *args, **kwargs: None

### CONSTANTS ###

ASPECTS = [
    "Hydration & Moisturization",
    "Anti-Aging & Skin Renewal",
    "Acne & Blemish Control",
    "Cleansing & Exfoliation",
    "Skin Sensitivity",
    "Skin Tone & Pigmentation",
    "Sun Protection",
    "Consistency & Texture",
    "Wear & Longevity",
    "Packaging & Size",
]

BASELINE_LABEL_MAP = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}
FINETUNE_LABELS    = ["positive", "negative", "neutral", "not_mentioned"]
LABEL2ID           = {l: i for i, l in enumerate(FINETUNE_LABELS)}
ID2LABEL           = {i: l for l, i in LABEL2ID.items()}

MODEL_NAME       = "yangheng/deberta-v3-base-absa-v1.1"
N_DEBERTA_LAYERS = 12


### DATASET ###

class ABSADataset(Dataset):
    def __init__(self, df, tokenizer,
                 text_col="cleaned_review_text",
                 label_col="label",
                 aspect_col="aspect",
                 max_length=512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.labels     = df[label_col].map(LABEL2ID).tolist()
        self.inputs     = [
            f"[ASPECT]{row[aspect_col]}[SEP]{row[text_col]}"
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.inputs[idx],
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


### DEFINE MODEL ###

def build_model(n_unfreeze: int):
    """Load pretrained model and replace the 3-class head with a 4-class head.

    n_unfreeze =  0    head + pooler only
    n_unfreeze =  N    top N transformer layers + head + pooler
    n_unfreeze = -1    full model (all layers)
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels              = len(FINETUNE_LABELS),
        id2label                = ID2LABEL,
        label2id                = LABEL2ID,
        ignore_mismatched_sizes = True,
    )

    if n_unfreeze == -1:
        print("Unfreezing full model")
        return model

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze head + pooler
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.pooler.parameters():
        param.requires_grad = True

    # Unfreeze top N transformer layers
    if n_unfreeze > 0:
        for layer in model.deberta.encoder.layer[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: ({100 * n_trainable / n_total:.1f}%)")

    return model


### OPTIMIZER ###

def get_optimizer(model, base_lr, decay_rate, head_lr_mult):
    """
    Assign different learning rates per layer
    1. Classification head + pooler : base_lr * head_lr_mult (highest, randomly initialised)
    2. Transformer layers           : base_lr * decay_rate^(distance from top) (lower = smaller LR)
    3. Everything else              : base_lr
    """

    head_params  = []
    layer_params = [[] for _ in range(N_DEBERTA_LAYERS)]
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "pooler" in name:
            head_params.append(param)
        else:
            matched = False
            for i in range(N_DEBERTA_LAYERS):
                if f"encoder.layer.{i}." in name:
                    layer_params[i].append(param)
                    matched = True
                    break
            if not matched:
                other_params.append(param)

    param_groups = [{"params": head_params, "lr": base_lr * head_lr_mult}]
    for i, params in enumerate(layer_params):
        if params:
            layer_lr = base_lr * (decay_rate ** (N_DEBERTA_LAYERS - i))
            param_groups.append({"params": params, "lr": layer_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})

    return torch.optim.AdamW(param_groups, weight_decay=0.01)


### EVALUATION ###

def evaluate(model, loader, device, criterion=None):
    model.eval()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    all_preds, all_labels  = [], []
    total_loss             = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs    = model(input_ids=input_ids, attention_mask=attention_mask)
            loss       = criterion(outputs.logits, labels)
            total_loss += loss.item()

            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    report   = classification_report(all_labels, all_preds,
                                     target_names=FINETUNE_LABELS, output_dict=True, zero_division=0)
    cm       = confusion_matrix(all_labels, all_preds)

    return avg_loss, report, cm, all_preds, all_labels


def print_metrics(report):

    for label in FINETUNE_LABELS:
        r = report[label]
        print(f"  {label:<20} Precision = {r['precision']:.3f}  Recall = {r['recall']:.3f}  "
              f"F1 = {r['f1-score']:.3f}  N = {int(r['support'])}")
        
    print(f"\n  Macro F1  : {report['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy  : {report['accuracy']:.4f}")


def plot_confusion_matrix(cm, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pd.DataFrame(cm, index=FINETUNE_LABELS, columns=FINETUNE_LABELS),
                annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


### BASELINE FUNCTION ###

def run_baseline(val_df,
                 text_col        = "cleaned_review_text",
                 label_col       = "label",
                 aspect_col      = "aspect",
                 output_dir      = "outputs",
                 threshold_start = 0.30,
                 threshold_stop  = 0.95,
                 threshold_step  = 0.05):
    """
    Baseline using the pretrained ABSA model.
    The not_mentioned class is determined by sweeping a threshold on the top class score for result where the model isn't confident enough.
    """

    from transformers import pipeline as hf_pipeline

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe       = hf_pipeline(
        "text-classification",
        model     = base_model,
        tokenizer = tokenizer,
        top_k     = None,
        truncation= True,
        max_length= 512,
        device    = 0 if torch.cuda.is_available() else -1,
    )

    rows = []
    for _, row in val_df.iterrows():
        aspect = row[aspect_col]
        text   = row[text_col]
        scores = pipe(f"[ASPECT]{aspect}[SEP]{text}")[0]
        sd     = {s["label"]: s["score"] for s in scores}
        rows.append({
            "true_label"    : row[label_col],
            "aspect"        : aspect,
            "score_positive": sd.get("Positive", 0),
            "score_negative": sd.get("Negative", 0),
            "score_neutral" : sd.get("Neutral",  0),
            "top_score"     : max(sd.values()),
            "top_label"     : max(sd, key=sd.get),
        })

    pred_df    = pd.DataFrame(rows)
    
    # Threshold sweep
    thresholds = np.arange(threshold_start, threshold_stop, threshold_step).round(2)
    sweep_rows = []

    for t in thresholds:
        preds = pred_df.apply(
            lambda r: "not_mentioned" if r["top_score"] < t
                      else BASELINE_LABEL_MAP[r["top_label"]], axis=1
        )
        f1 = f1_score(pred_df["true_label"], preds, average="macro",
                      labels=FINETUNE_LABELS, zero_division=0)
        sweep_rows.append({"threshold": t, "macro_f1": round(f1, 4)})

    sweep_df = pd.DataFrame(sweep_rows)
    best_t   = sweep_df.loc[sweep_df["macro_f1"].idxmax(), "threshold"]
    print(f"Best threshold: {best_t}  (F1 macro={sweep_df['macro_f1'].max():.4f})")

    final_preds = pred_df.apply(
        lambda r: "not_mentioned" if r["top_score"] < best_t
                  else BASELINE_LABEL_MAP[r["top_label"]], axis=1
    ).tolist()

    report = classification_report(pred_df["true_label"].tolist(), final_preds,
                                   labels=FINETUNE_LABELS, output_dict=True, zero_division=0)
    cm     = confusion_matrix(pred_df["true_label"].tolist(), final_preds,
                              labels=FINETUNE_LABELS)

    print_metrics(report)
    plot_confusion_matrix(cm, title=f"Baseline  threshold={best_t}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred_df["pred_label"] = final_preds

    return pred_df, sweep_df


### FINE-TUNE FUNCTION ###

def compute_class_weights(train_df, label_col='label'):
    label_counts = train_df[label_col].value_counts()
    total        = len(train_df)

    weights = torch.tensor([
        np.sqrt(total / (len(FINETUNE_LABELS) * (label_counts[label] if label in label_counts else 1)))
        for label in FINETUNE_LABELS
    ], dtype=torch.float)
    
    return weights
    
def run_finetune(train_df, val_df,
                 text_col     = "cleaned_review_text",
                 label_col    = "label",
                 aspect_col   = "aspect",
                 output_dir   = "outputs",
                 n_unfreeze   = 0,
                 epochs       = 10,
                 batch_size   = 16,
                 grad_accum   = 2,
                 lr           = 2e-5,
                 decay_rate   = 0.9,
                 head_lr_mult = 10.0,
                 patience     = 3):
    """Fine-tune the pretrained ABSA model on your labelled data"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n FINE TUNING: n_unfreeze={n_unfreeze} | device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = build_model(n_unfreeze).to(device)

    train_ds = ABSADataset(train_df, tokenizer, text_col=text_col, label_col=label_col, aspect_col=aspect_col)
    val_ds   = ABSADataset(val_df, tokenizer, text_col=text_col, label_col=label_col, aspect_col=aspect_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    optimizer   = get_optimizer(model, base_lr=lr, decay_rate=decay_rate, head_lr_mult=head_lr_mult)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer,
                                                  num_warmup_steps   = int(0.1 * total_steps),
                                                  num_training_steps = total_steps)

    class_weights = compute_class_weights(train_df).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_val_loss  = float("inf")
    patience_count = 0
    history        = []
    ckpt_path      = out / f"best_model_unfreeze{n_unfreeze}"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels) / grad_accum
            loss.backward()

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum

        avg_train_loss           = train_loss / len(train_loader)
        val_loss, report, cm, _, _ = evaluate(model, val_loader, device, criterion=criterion)
        macro_f1                 = report["macro avg"]["f1-score"]

        print(f"Epoch {epoch}/{epochs}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"macro_F1={macro_f1:.4f}")

        history.append({"epoch": epoch, "train_loss": avg_train_loss,
                        "val_loss": val_loss, "macro_f1": macro_f1})

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            hf_logging.disable_progress_bar()
            model.save_pretrained(ckpt_path)
            hf_logging.enable_progress_bar()
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Saved best checkpoint to {ckpt_path}")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print("  Early stopping.")
                break

    print(f"\nLoading best checkpoint for final evaluation")
    best_model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
    _, report, cm, _, _ = evaluate(best_model, val_loader, device, criterion=criterion)
    print_metrics(report)
    plot_confusion_matrix(cm, title=f"Fine-tune  n_unfreeze={n_unfreeze}")

    return best_model, pd.DataFrame(history)