import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

### CONSTANTS ###

DETECTION_LABELS   = ["not_mentioned", "present"]
DETECTION_ID2LABEL = {i: l for i, l in enumerate(DETECTION_LABELS)}

SENTIMENT_LABELS   = ["negative", "neutral", "positive"]

TEXT_COL   = "cleaned_demojize_review_text"
ASPECT_COL = "aspect"

OUTPUT_COLS = [
    "review_id", "aspect", "year", "main_topic", "rating",
    "cleaned_review_text", "cleaned_demojize_review_text", "cleaned_review_title",
    "det_pred", "sent_pred", "final_pred",
]

### DETECTION DATASET (no labels) ###

class DetectionInferenceDataset(Dataset):
    """
    Define Dataset for inference for aspect detection. 
    Replicates the input format used for training in DetectionDataset class."""

    def __init__(self, df, tokenizer, text_col=TEXT_COL, aspect_col=ASPECT_COL,
                 max_length=512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.pairs = [(row[aspect_col], row[text_col]) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        aspect, text = self.pairs[idx]
        enc = self.tokenizer(
            f"Does this review mention {aspect}?",
            str(text),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def run_detection(df, ckpt_path, device, batch_size=32, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path, use_safetensors=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    ds = DetectionInferenceDataset(df, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    preds = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [DETECTION_ID2LABEL[p] for p in preds]


def run_sentiment_setfit(df, ckpt_path, text_col=TEXT_COL, aspect_col=ASPECT_COL,
                          batch_size=32):
    """
    Filtered dataframe to rows where det_pred == 'present'.
    Replicates the input format in training."""

    from setfit import SetFitModel

    if len(df) == 0:
        return []

    model = SetFitModel.from_pretrained(ckpt_path)

    inputs = [
        f"[ASPECT]{row[aspect_col]}[SEP]{row[text_col]}"
        for _, row in df.iterrows()
    ]

    preds = model.predict(inputs, batch_size=batch_size)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    if len(preds) > 0 and isinstance(preds[0], str):
        return list(preds)
    
    return [SENTIMENT_LABELS[int(p)] for p in preds]


def run_cascade_inference(input_csv, output_csv, det_ckpt, setfit_ckpt,
                           batch_size=32, max_length=512):
    """Main function to run cascade inference."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    df = pd.read_csv(input_csv)
    missing = [c for c in ["review_id", ASPECT_COL, TEXT_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
 
    df = df.reset_index(drop=True)
 
    # Classifier 1: Aspect detection
    print(f"\nRunning detection: {len(df)} rows | checkpoint={det_ckpt}")
    df["det_pred"] = run_detection(
        df, det_ckpt, device,
        batch_size=batch_size, max_length=max_length,
    )
    print(df["det_pred"].value_counts())
 
    # Classifier 2: Sentiment (only rows predicted "present")
    present_mask = df["det_pred"] == "present"
    present_df = df[present_mask].copy()
    print(f"\nRunning sentiment (SetFit): {len(present_df)} rows | "
          f"checkpoint={setfit_ckpt}")
 
    sent_preds = run_sentiment_setfit(present_df, setfit_ckpt, batch_size=batch_size)
 
    df["sent_pred"] = None
    df.loc[present_mask, "sent_pred"] = sent_preds
 
    # Final label: sentiment where present, else not_mentioned
    df["final_pred"] = np.where(
        df["det_pred"] == "present", df["sent_pred"], "not_mentioned"
    )
 
    print("\nLabel distribution:")
    print(df["final_pred"].value_counts())
 
    out_cols = [c for c in OUTPUT_COLS if c in df.columns]
    df[out_cols].to_csv(output_csv, index=False)
    print(f"\nSaved predictions to {output_csv}")
 
    return df[out_cols]