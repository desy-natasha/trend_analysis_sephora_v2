import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup, pipeline as hf_pipeline)
from transformers import logging as hf_logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Suppress warnings
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
warnings.filterwarnings("ignore", message="Precision is ill-defined", category=UserWarning)
os.environ["TQDM_DISABLE"] = "1"

### CONSTANTS ###

DETECTION_LABELS   = ["not_mentioned", "present"]
DETECTION_LABEL2ID = {l: i for i, l in enumerate(DETECTION_LABELS)}
DETECTION_ID2LABEL = {i: l for l, i in DETECTION_LABEL2ID.items()}

SENTIMENT_LABELS   = ["negative", "neutral", "positive"]
SENTIMENT_LABEL2ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID2LABEL = {i: l for l, i in SENTIMENT_LABEL2ID.items()}

# Final output labels to evaluate the full cascade
FINAL_LABELS = ["positive", "negative", "neutral", "not_mentioned"]

BACKBONE_CONFIG = {
    # 1. Aspect Detection - BERT-based models
    "bert-base-uncased"                    : ("bert",    12),
    "roberta-base"                         : ("roberta", 12),
    "microsoft/deberta-v3-base"            : ("deberta", 12),

    # 2.1. Sentiment Analysis - BERT-based models
    "yangheng/deberta-v3-base-absa-v1.1"   : ("deberta", 12),
}

# 2.2. Sentiment Analysis - NLI models for zero-shot approach
NLI_MODELS = [
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "cross-encoder/nli-roberta-base",
    "facebook/bart-large-mnli",
]

# 2.3. Sentiment Analysis - SetFit models for few-shot approach
SETFIT_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-roberta-large-v1",
]

### HELPERS ###

def to_detection_label(label: str) -> str:
    return "not_mentioned" if label == "not_mentioned" else "present"

def to_detection_id(label: str) -> int:
    return DETECTION_LABEL2ID[to_detection_label(label)]

def get_encoder_attr(model, backbone: str):
    """Return the encoder sub-module for a given backbone."""

    attr_name = BACKBONE_CONFIG[backbone][0]
    if not hasattr(model, attr_name):
        raise ValueError(
            f"Model has no attribute '{attr_name}'. "
            f"Check BACKBONE_CONFIG for backbone='{backbone}'."
        )
    return getattr(model, attr_name)

def get_n_layers(backbone: str) -> int:
    return BACKBONE_CONFIG[backbone][1]

### DATASET ###

class DetectionDataset(Dataset):
    """Detection dataset using all rows. Labels includes present and not_mentioned."""

    def __init__(self, df, tokenizer,
                 backbone="microsoft/deberta-v3-base",
                 text_col="cleaned_review_text",
                 label_col="label",
                 aspect_col="aspect",
                 max_length=512):
        self.tokenizer  = tokenizer
        self.backbone   = backbone
        self.max_length = max_length
        self.labels = [to_detection_id(l) for l in df[label_col]]
        self.pairs  = [(row[aspect_col], row[text_col]) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        aspect, text = self.pairs[idx]

        if "absa" in self.backbone.lower():
            # Format for Yangheng ABSA
            enc = self.tokenizer(
                f"[ASPECT]{aspect}[SEP]{text}",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            # Standard format for bert/roberta/deberta-v3-base
            enc = self.tokenizer(
                f"Does this review mention {aspect}?",
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

class SentimentDataset(Dataset):
    """Sentiment dataset using only present rows. Labels includes positive, neutral, negative."""

    def __init__(self, df, tokenizer,
                 text_col="cleaned_review_text",
                 label_col="label",
                 aspect_col="aspect",
                 max_length=512):
        present_df = df[df[label_col] != "not_mentioned"].reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.labels = [SENTIMENT_LABEL2ID[l] for l in present_df[label_col]]
        self.inputs = [
            f"[ASPECT]{row[aspect_col]}[SEP]{row[text_col]}"
            for _, row in present_df.iterrows()
        ]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

### DEFINE MODEL ###

def build_model(backbone: str,
                num_labels: int,
                id2label: dict,
                label2id: dict,
                n_unfreeze: int = 0,
                ignore_mismatched_sizes: bool = False):
    """Load pretrained model and replace the classification head.

    n_unfreeze =  0    head + pooler only
    n_unfreeze =  N    top N transformer layers + head + pooler
    n_unfreeze = -1    full model (all layers)
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels              = num_labels,
        id2label                = id2label,
        label2id                = label2id,
        ignore_mismatched_sizes = ignore_mismatched_sizes,
        use_safetensors         = True,
        torch_dtype             = torch.float32,
    )

    if n_unfreeze == -1:
        print(f"  Unfreezing full model")
        n_trainable = sum(p.numel() for p in model.parameters())
        n_total     = n_trainable
        return model

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze classification head
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Always unfreeze pooler
    if hasattr(model, "pooler") and model.pooler is not None:
        for param in model.pooler.parameters():
            param.requires_grad = True

    # Unfreeze top N transformer layers
    if n_unfreeze > 0:
        encoder = get_encoder_attr(model, backbone)
        for layer in encoder.encoder.layer[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {100 * n_trainable / n_total:.1f}%")

    return model

### OPTIMIZER ###

def get_optimizer(model, backbone: str,
                  base_lr: float      = 2e-5,
                  decay_rate: float   = 0.9,
                  head_lr_mult: float = 10.0):
    """
    Assign different learning rates per layer
    1. Classification head + pooler : base_lr * head_lr_mult (highest, randomly initialised)
    2. Transformer layers           : base_lr * decay_rate^(distance from top) (lower = smaller LR)
    3. Everything else              : base_lr
    """
    n_layers     = get_n_layers(backbone)
    head_params  = []
    layer_params = [[] for _ in range(n_layers)]
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "pooler" in name:
            head_params.append(param)
        else:
            matched = False
            for i in range(n_layers):
                if f"encoder.layer.{i}." in name:
                    layer_params[i].append(param)
                    matched = True
                    break
            if not matched:
                other_params.append(param)

    param_groups = [{"params": head_params, "lr": base_lr * head_lr_mult}]
    for i, params in enumerate(layer_params):
        if params:
            layer_lr = base_lr * (decay_rate ** (n_layers - i))
            param_groups.append({"params": params, "lr": layer_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})

    return torch.optim.AdamW(param_groups, weight_decay=0.01)

### CLASS WEIGHTS ###

def compute_class_weights_detection(train_df, label_col="label"):
    binary_labels = [to_detection_label(l) for l in train_df[label_col]]
    counts        = pd.Series(binary_labels).value_counts()
    total         = len(binary_labels)
    weights = torch.tensor([
        np.sqrt(total / (len(DETECTION_LABELS) * (counts.get(l, 1))))
        for l in DETECTION_LABELS
    ], dtype=torch.float)
    return weights

def compute_class_weights_sentiment(train_df, label_col="label"):
    present_df = train_df[train_df[label_col] != "not_mentioned"]
    counts     = present_df[label_col].value_counts()
    total      = len(present_df)
    weights = torch.tensor([
        np.sqrt(total / (len(SENTIMENT_LABELS) * (counts.get(l, 1))))
        for l in SENTIMENT_LABELS
    ], dtype=torch.float)
    return weights

### EVALUATION ###

def evaluate_detection(model, loader, device, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            outputs    = model(input_ids      = batch["input_ids"].to(device),
                               attention_mask = batch["attention_mask"].to(device))
            loss       = criterion(outputs.logits, batch["label"].to(device))
            total_loss += loss.item()
            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())
    avg_loss = total_loss / len(loader)
    report   = classification_report(all_labels, all_preds,
                                     target_names=DETECTION_LABELS,
                                     output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, report, cm

def evaluate_sentiment(model, loader, device, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            outputs    = model(input_ids      = batch["input_ids"].to(device),
                               attention_mask = batch["attention_mask"].to(device))
            loss       = criterion(outputs.logits, batch["label"].to(device))
            total_loss += loss.item()
            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())
    avg_loss = total_loss / len(loader)
    report   = classification_report(all_labels, all_preds,
                                     target_names=SENTIMENT_LABELS,
                                     output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, report, cm

def evaluate_cascade(detection_model, sentiment_model,
                     detection_tokenizer, sentiment_tokenizer,
                     val_df, device,
                     det_backbone="microsoft/deberta-v3-base",
                     text_col   = "cleaned_review_text",
                     label_col  = "label",
                     aspect_col = "aspect",
                     batch_size = 32,
                     max_length = 512):
    """
    Evaluate the cascaded model on the validation set.
    The result from first classifier will fed into the second classifier to evaluate the full cascade performance.
    """
    detection_model.eval()
    sentiment_model.eval()

    true_labels = val_df[label_col].tolist()
    final_preds = ["not_mentioned"] * len(val_df)

    # Stage 1: Aspect detection
    det_dataset = DetectionDataset(val_df, detection_tokenizer, backbone=det_backbone,
                                   text_col=text_col, label_col=label_col,
                                   aspect_col=aspect_col, max_length=max_length)
    det_loader  = DataLoader(det_dataset, batch_size=batch_size, shuffle=False)

    detection_preds = []
    with torch.no_grad():
        for batch in det_loader:
            outputs = detection_model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
            )
            detection_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())

    # Stage 2: Sentiment classification on samples that are predicted as present only
    present_indices = [i for i, p in enumerate(detection_preds)
                       if DETECTION_ID2LABEL[p] == "present"]

    if present_indices:
        present_df = val_df.iloc[present_indices].reset_index(drop=True)

        temp_df = present_df.copy()
        temp_df[label_col] = temp_df[label_col].replace("not_mentioned", "negative")

        sent_dataset = SentimentDataset(temp_df, sentiment_tokenizer,
                                        text_col=text_col, label_col=label_col,
                                        aspect_col=aspect_col, max_length=max_length)
        sent_loader  = DataLoader(sent_dataset, batch_size=batch_size, shuffle=False)

        sentiment_preds = []
        with torch.no_grad():
            for batch in sent_loader:
                outputs = sentiment_model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                )
                sentiment_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())

        for list_pos, df_idx in enumerate(present_indices):
            final_preds[df_idx] = SENTIMENT_ID2LABEL[sentiment_preds[list_pos]]

    report = classification_report(true_labels, final_preds,
                                   labels=FINAL_LABELS,
                                   output_dict=True, zero_division=0)
    cm     = confusion_matrix(true_labels, final_preds, labels=FINAL_LABELS)
    return report, cm, final_preds

### METRICS AND PLOTS ###

def print_detection_metrics(report, backbone: str):
    for label in DETECTION_LABELS:
        r = report[label]
        print(f"  {label:<20} P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  F1={r['f1-score']:.3f}  N={int(r['support'])}")
    print(f"\n  Macro F1 : {report['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy : {report['accuracy']:.4f}")

def print_sentiment_metrics(report, model_name: str):
    for label in SENTIMENT_LABELS:
        r = report[label]
        print(f"  {label:<20} P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  F1={r['f1-score']:.3f}  N={int(r['support'])}")
    print(f"\n  Macro F1 : {report['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy : {report['accuracy']:.4f}")

def plot_confusion_matrix(cm, labels, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pd.DataFrame(cm, index=labels, columns=labels),
                annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

def print_cascade_metrics(report):
    for label in FINAL_LABELS:
        r = report[label]
        print(f"  {label:<20} P={r['precision']:.3f}  "
              f"R={r['recall']:.3f}  F1={r['f1-score']:.3f}  N={int(r['support'])}")
    print(f"\n  Macro F1 : {report['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy : {report['accuracy']:.4f}")

### TRAINING LOOP ###

def _train_loop(model, train_loader, val_loader,
                optimizer, scheduler, criterion,
                evaluate_fn, device,
                epochs, grad_accum, patience, ckpt_path):

    best_val_loss  = float("inf")
    best_epoch     = 0
    best_macro_f1 = 0.0
    patience_count = 0
    history        = []

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

        avg_train_loss       = train_loss / len(train_loader)
        val_loss, report, cm = evaluate_fn(model, val_loader, device, criterion)
        macro_f1             = report["macro avg"]["f1-score"]

        print(f"  Epoch {epoch}/{epochs}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"macro_F1={macro_f1:.4f}")

        history.append({"epoch": epoch, "train_loss": avg_train_loss,
                        "val_loss": val_loss, "macro_f1": macro_f1})

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            hf_logging.disable_progress_bar()
            model.save_pretrained(ckpt_path, safe_serialization=True)
            hf_logging.enable_progress_bar()
            # print(f"  Saved best checkpoint to {ckpt_path}")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print("  Early stopping.")
                break

    print(f"\nLoading best checkpoint from epoch {best_epoch} with val_loss={best_val_loss:.4f}")

    return pd.DataFrame(history)

### STAGE 1: ASPECT DETECTION ABLATION ###

def run_detection_ablation(train_df, val_df,
                            backbone     = "microsoft/deberta-v3-base",
                            n_unfreeze   = 3,
                            text_col     = "cleaned_review_text",
                            label_col    = "label",
                            aspect_col   = "aspect",
                            output_dir   = "outputs",
                            epochs       = 10,
                            batch_size   = 16,
                            grad_accum   = 2,
                            lr           = 2e-5,
                            decay_rate   = 0.9,
                            head_lr_mult = 10.0,
                            warmup_ratio = 0.1,
                            patience     = 3,
                            max_length   = 512):
    """Fine-tuned the pretrained model for aspect detection."""

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_name = backbone.replace("/", "_")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if backbone == "yangheng/deberta-v3-base-absa-v1.1":
        model     = build_model(backbone,
                                 num_labels = len(DETECTION_LABELS),
                                 id2label   = DETECTION_ID2LABEL,
                                 label2id   = DETECTION_LABEL2ID,
                                 n_unfreeze = n_unfreeze,
                                 ignore_mismatched_sizes=True).to(device)
    else:
        model     = build_model(backbone,
                             num_labels = len(DETECTION_LABELS),
                             id2label   = DETECTION_ID2LABEL,
                             label2id   = DETECTION_LABEL2ID,
                             n_unfreeze = n_unfreeze).to(device)

    train_ds = DetectionDataset(train_df, tokenizer, backbone=backbone, text_col=text_col,
                                label_col=label_col, aspect_col=aspect_col,
                                max_length=max_length)
    val_ds   = DetectionDataset(val_df,   tokenizer, backbone=backbone, text_col=text_col,
                                label_col=label_col, aspect_col=aspect_col,
                                max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size,     shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    optimizer   = get_optimizer(model, backbone, base_lr=lr,
                                decay_rate=decay_rate, head_lr_mult=head_lr_mult)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(warmup_ratio * total_steps),
        num_training_steps = total_steps,
    )

    class_weights = compute_class_weights_detection(train_df, label_col).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    ckpt_path = Path(output_dir) / f"detection_{safe_name}_unfreeze{n_unfreeze}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    history = _train_loop(model, train_loader, val_loader,
                           optimizer, scheduler, criterion,
                           evaluate_detection, device,
                           epochs, grad_accum, patience, ckpt_path)

    best_model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path, use_safetensors=True, torch_dtype=torch.float32
    ).to(device)
    tokenizer.save_pretrained(ckpt_path)

    _, report, cm = evaluate_detection(best_model, val_loader, device, criterion)
    print_detection_metrics(report, backbone)
    plot_confusion_matrix(
        cm, DETECTION_LABELS,
        title=f"Detection | {backbone} | unfreeze={n_unfreeze}"
    )

    return {
        "model"     : best_model,
        "tokenizer" : tokenizer,
        "history"   : history,
        "report"    : report,
        "cm"        : cm,
        "backbone"  : backbone,
        "n_unfreeze": n_unfreeze,
        "ckpt_path" : str(ckpt_path),
    }

### STAGE 2.1: SENTIMENT ANALYSIS - BERT-BASED ABLATION ###

def run_sentiment_ablation(train_df, val_df,
                            backbone     = "yangheng/deberta-v3-base-absa-v1.1",
                            n_unfreeze   = 3,
                            text_col     = "cleaned_review_text",
                            label_col    = "label",
                            aspect_col   = "aspect",
                            output_dir   = "outputs",
                            epochs       = 10,
                            batch_size   = 16,
                            grad_accum   = 2,
                            lr           = 2e-5,
                            decay_rate   = 0.9,
                            head_lr_mult = 10.0,
                            patience     = 3,
                            max_length   = 512):
    """Fine-tuned ABSA sentiment classifier for sentiment classification.."""

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_name = backbone.replace("/", "_")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model     = build_model(backbone,
                             num_labels              = len(SENTIMENT_LABELS),
                             id2label                = SENTIMENT_ID2LABEL,
                             label2id                = SENTIMENT_LABEL2ID,
                             n_unfreeze              = n_unfreeze,
                             ignore_mismatched_sizes = False,
                             ).to(device)

    train_ds = SentimentDataset(train_df, tokenizer, text_col=text_col,
                                label_col=label_col, aspect_col=aspect_col,
                                max_length=max_length)
    val_ds   = SentimentDataset(val_df,   tokenizer, text_col=text_col,
                                label_col=label_col, aspect_col=aspect_col,
                                max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size,     shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    optimizer   = get_optimizer(model, backbone, base_lr=lr,
                                decay_rate=decay_rate, head_lr_mult=head_lr_mult)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(0.1 * total_steps),
        num_training_steps = total_steps,
    )

    class_weights = compute_class_weights_sentiment(train_df, label_col).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    ckpt_path = Path(output_dir) / f"sentiment_{safe_name}_unfreeze{n_unfreeze}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    history = _train_loop(model, train_loader, val_loader,
                           optimizer, scheduler, criterion,
                           evaluate_sentiment, device,
                           epochs, grad_accum, patience, ckpt_path)

    best_model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path, use_safetensors=True, torch_dtype=torch.float32
    ).to(device)
    tokenizer.save_pretrained(ckpt_path)

    _, report, cm = evaluate_sentiment(best_model, val_loader, device, criterion)
    print_sentiment_metrics(report, backbone)
    plot_confusion_matrix(
        cm, SENTIMENT_LABELS,
        title=f"Sentiment (Fine-tuned) | {backbone} | unfreeze={n_unfreeze}"
    )

    return {
        "approach"  : "finetuned",
        "model"     : best_model,
        "tokenizer" : tokenizer,
        "history"   : history,
        "report"    : report,
        "cm"        : cm,
        "backbone"  : backbone,
        "n_unfreeze": n_unfreeze,
        "ckpt_path" : str(ckpt_path),
    }

### STAGE 2.2: SENTIMENT ANALYSIS - NLI MODELS ABLATION ###

# Hypothesis template used for zero-shot ABSA.
# The aspect is embedded in the sequence; {} is replaced by the candidate label.
_NLI_HYPOTHESIS = "The sentiment expressed towards the mentioned aspect is {}."

def run_nli_sentiment(val_df,
                      model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                      text_col   = "cleaned_review_text",
                      label_col  = "label",
                      aspect_col = "aspect",
                      batch_size = 32,
                      hypothesis_template: str = _NLI_HYPOTHESIS):
    """Evaluate zero-shot ABSA sentiment using an NLI model."""

    print(f"\nNLI: {model_name}")

    device_id = 0 if torch.cuda.is_available() else -1
    classifier = hf_pipeline(
        "zero-shot-classification",
        model     = model_name,
        device    = device_id,
        batch_size = batch_size,
    )

    present_df = val_df[val_df[label_col] != "not_mentioned"].reset_index(drop=True)

    # Embed the aspect directly into the sequence to provide explicit aspect information to the NLI model.
    sequences = [
        f"Aspect: {row[aspect_col]}. Review: {row[text_col]}"
        for _, row in present_df.iterrows()
    ]
    true_labels = [SENTIMENT_LABEL2ID[l] for l in present_df[label_col]]

    results = classifier(
        sequences,
        candidate_labels  = SENTIMENT_LABELS,
        hypothesis_template = hypothesis_template,
        multi_label       = False,
    )

    if isinstance(results, dict):
        results = [results]

    all_preds = [SENTIMENT_LABEL2ID[r["labels"][0]] for r in results]

    report = classification_report(
        true_labels, all_preds,
        target_names = SENTIMENT_LABELS,
        output_dict  = True,
        zero_division = 0,
    )
    cm = confusion_matrix(true_labels, all_preds)

    print_sentiment_metrics(report, model_name)
    plot_confusion_matrix(
        cm, SENTIMENT_LABELS,
        title=f"Sentiment (NLI zero-shot) | {model_name}"
    )

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "approach"  : "nli",
        "backbone"  : model_name,
        "report"    : report,
        "cm"        : cm,
        "n_unfreeze": None,  
    }

### STAGE 2.3: SENTIMENT ANALYSIS - SETFIT SENTIMENT ABLATION ###

def run_setfit_sentiment(train_df, val_df,
                         model_name        = "sentence-transformers/all-mpnet-base-v2",
                         text_col          = "cleaned_review_text",
                         label_col         = "label",
                         aspect_col        = "aspect",
                         num_iterations    = 20,
                         num_epochs        = 1,
                         batch_size        = 16,
                         samples_per_label = None,
                         output_dir        = "outputs"):
    """Fine-tuned SetFit model for sentiment classification."""
    
    try:
        from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments
        from datasets import Dataset as HFDataset
    except ImportError as e:
        raise ImportError(
            "SetFit is not installed. Run: pip install setfit datasets"
        ) from e

    print(f"\nSetFit: {model_name}")

    safe_name = model_name.replace("/", "_")
    ckpt_path = Path(output_dir) / f"setfit_{safe_name}"

    present_train = (train_df[train_df[label_col] != "not_mentioned"]
                     .reset_index(drop=True).copy())
    present_val   = (val_df[val_df[label_col]   != "not_mentioned"]
                     .reset_index(drop=True).copy())

    # Aspect-aware input format (mirrors fine-tuned approach)
    def make_text(row):
        return f"[ASPECT]{row[aspect_col]}[SEP]{row[text_col]}"

    present_train["text"]  = present_train.apply(make_text, axis=1)
    present_val["text"]    = present_val.apply(make_text, axis=1)
    present_train["label"] = present_train[label_col].map(SENTIMENT_LABEL2ID)
    present_val["label"]   = present_val[label_col].map(SENTIMENT_LABEL2ID)

    if samples_per_label is not None:
        present_train = (
            present_train
            .groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(samples_per_label, len(g)), random_state=42))
            .reset_index(drop=True)
        )
        print(f"  Sub-sampled training set: {len(present_train)} rows "
              f"({samples_per_label} per label)")

    train_dataset = HFDataset.from_pandas(present_train[["text", "label"]])
    val_dataset   = HFDataset.from_pandas(present_val[["text", "label"]])

    model = SetFitModel.from_pretrained(
        model_name,
        num_classes = len(SENTIMENT_LABELS),
        labels      = SENTIMENT_LABELS,
    )

    args = TrainingArguments(
        num_iterations = num_iterations,
        num_epochs     = num_epochs,
        batch_size     = batch_size,
        seed           = 42,
    )

    trainer = SetFitTrainer(
        model          = model,
        args           = args,
        train_dataset  = train_dataset,
        eval_dataset   = val_dataset,
        metric         = "accuracy",
    )
    trainer.train()

    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_path))
    print(f"  Saved checkpoint to {ckpt_path}")

    preds       = model.predict(present_val["text"].tolist())
    true_labels = present_val["label"].tolist()

    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    if len(preds) > 0 and isinstance(preds[0], str):
        preds = [SENTIMENT_LABEL2ID[p] for p in preds]
    preds = [int(p) for p in preds]

    report = classification_report(
        true_labels, preds,
        target_names  = SENTIMENT_LABELS,
        output_dict   = True,
        zero_division = 0,
    )
    cm = confusion_matrix(true_labels, preds)

    print_sentiment_metrics(report, model_name)
    plot_confusion_matrix(
        cm, SENTIMENT_LABELS,
        title=f"Sentiment (SetFit) | {model_name}"
    )

    return {
        "approach"         : "setfit",
        "backbone"         : model_name,
        "model"            : model,
        "report"           : report,
        "cm"               : cm,
        "n_unfreeze"       : None,   
        "num_iterations"   : num_iterations,
        "samples_per_label": samples_per_label,
    }

### FINAL CASCADE LOOP ###

def run_cascade_ablation(train_df, val_df,
                          det_backbone    = "microsoft/deberta-v3-base",
                          sent_backbone   = "yangheng/deberta-v3-base-absa-v1.1",
                          det_n_unfreeze  = 3,
                          sent_n_unfreeze = 3,
                          text_col        = "cleaned_review_text",
                          label_col       = "label",
                          aspect_col      = "aspect",
                          output_dir      = "outputs",
                          # Detection hyperparams
                          det_epochs      = 10,
                          det_batch_size  = 16,
                          det_lr          = 2e-5,
                          det_patience    = 3,
                          # Sentiment hyperparams
                          sent_epochs     = 10,
                          sent_batch_size = 16,
                          sent_lr         = 2e-5,
                          sent_patience   = 3,
                          # Both classifiers
                          grad_accum      = 2,
                          decay_rate      = 0.9,
                          head_lr_mult    = 10.0,
                          max_length      = 512):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1: Aspect detection ablation
    print(f"ASPECT DETECTION | backbone={det_backbone} | unfreeze={det_n_unfreeze}")

    det_result = run_detection_ablation(
        train_df     = train_df,
        val_df       = val_df,
        backbone     = det_backbone,
        n_unfreeze   = det_n_unfreeze,
        text_col     = text_col,
        label_col    = label_col,
        aspect_col   = aspect_col,
        output_dir   = output_dir,
        epochs       = det_epochs,
        batch_size   = det_batch_size,
        grad_accum   = grad_accum,
        lr           = det_lr,
        decay_rate   = decay_rate,
        head_lr_mult = head_lr_mult,
        patience     = det_patience,
        max_length   = max_length,
    )

    # Stage 2: Sentiment ablation (ground truth present)

    print(f"SENTIMENT CLASSIFIER | backbone={sent_backbone} | unfreeze={sent_n_unfreeze}")

    sent_result = run_sentiment_ablation(
        train_df     = train_df,
        val_df       = val_df,
        backbone     = sent_backbone,
        n_unfreeze   = sent_n_unfreeze,
        text_col     = text_col,
        label_col    = label_col,
        aspect_col   = aspect_col,
        output_dir   = output_dir,
        epochs       = sent_epochs,
        batch_size   = sent_batch_size,
        grad_accum   = grad_accum,
        lr           = sent_lr,
        decay_rate   = decay_rate,
        head_lr_mult = head_lr_mult,
        patience     = sent_patience,
        max_length   = max_length,
    )

    # Final cascade evaluation
    print(f"CASCADE EVALUATION | det={det_backbone} | sent={sent_backbone}")

    report, cm, final_preds = evaluate_cascade(
        det_result["model"],     sent_result["model"],
        det_result["tokenizer"], sent_result["tokenizer"],
        val_df, device,
        det_backbone = det_backbone,
        text_col   = text_col,
        label_col  = label_col,
        aspect_col = aspect_col,
        batch_size = max(det_batch_size, sent_batch_size) * 2,
        max_length = max_length,
    )
    print_cascade_metrics(report)
    plot_confusion_matrix(
        cm, FINAL_LABELS,
        title=f"Cascade | det={det_backbone} | sent={sent_backbone}"
    )

    val_df = val_df.copy()
    val_df["pred_label"] = final_preds

    return {
        "det_backbone"     : det_backbone,
        "sent_backbone"    : sent_backbone,
        "det_n_unfreeze"   : det_n_unfreeze,
        "sent_n_unfreeze"  : sent_n_unfreeze,
        "detection_result" : det_result,
        "sentiment_result" : sent_result,
        "cascade_report"   : report,
        "cascade_cm"       : cm,
        "val_predictions"  : val_df,
    }
