import argparse
import random
import warnings
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

warnings.filterwarnings("ignore")


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

LABEL_MAP = {
    "Positive": "positive",
    "Negative": "negative",
    "Neutral":  "neutral",
}


def load_model():
    """Load yangheng/deberta-v3-base-absa-v1.1 from HuggingFace."""

    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,  # return all labels
        truncation=True,
        max_length=512,
    )


def predict_single(pipe, review, aspect, threshold = 0.5):
    """Formatted input and return aspect, sentiment, and confidence scores."""

    scores = pipe(f"[ASPECT]{aspect}[SEP]{review}")[0]
    score_dict = {s["label"]: s["score"] for s in scores}

    top_label = max(score_dict, key=score_dict.get)
    top_score = score_dict[top_label]

    # Initial approach only assign not_mentioned if score is below threshold for highest scoring
    # This shows that the model is not confident enough to assign any sentiment, so we treat it as not mentioned
    not_mentioned = top_score < threshold

    return {
        "aspect":         aspect,
        "sentiment":      "not_mentioned" if not_mentioned else LABEL_MAP[top_label],
        "confidence":     round(top_score, 4),
        "score_positive": round(score_dict.get("Positive", 0), 4),
        "score_negative": round(score_dict.get("Negative", 0), 4),
        "score_neutral":  round(score_dict.get("Neutral",  0), 4),
        "not_mentioned":  not_mentioned,
    }


def run_inference(pipe, reviews, threshold = 0.5):
    """Loop through all reviews and aspects"""
    results = []
    total = len(reviews) * len(ASPECTS)

    for idx, review in enumerate(reviews):
        for aspect in ASPECTS:
            pred = predict_single(pipe, review, aspect, threshold)
            pred["review_id"]   = idx
            pred["review_text"] = review
            results.append(pred)

    return pd.DataFrame(results)

def summarise_results(df):
    """Aggregate sentiment counts and mention rates per aspect"""
    total_reviews = df["review_id"].nunique()
    mentioned = df[~df["not_mentioned"]].copy()

    summary = (
        mentioned
        .groupby(["aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    mention_counts = (
        mentioned
        .groupby("aspect")["review_id"]
        .nunique()
        .reset_index()
        .rename(columns={"review_id": "reviews_mentioning"})
    )
    summary = summary.merge(mention_counts, on="aspect")
    summary["mention_rate"] = (summary["reviews_mentioning"] / total_reviews * 100).round(1)

    for col in ["positive", "negative", "neutral"]:
        if col not in summary.columns:
            summary[col] = 0

    total_mentioned = summary[["positive", "negative", "neutral"]].sum(axis=1)
    for col in ["positive", "negative", "neutral"]:
        summary[f"{col}"] = (summary[col] / total_mentioned * 100).round(1)

    return summary.sort_values("mention_rate", ascending=False)


def threshold_sensitivity(df, thresholds):
    """Experiment with different confidence thresholds to see how mention rates change"""
    rows = []
    for t in thresholds:
        n_mentioned = (df["confidence"] >= t).sum()
        rows.append({
            "threshold":      t,
            "mentioned":      n_mentioned,
            "not_mentioned":  len(df) - n_mentioned,
            "mention_rate": round(n_mentioned / len(df) * 100, 1),
        })
    return pd.DataFrame(rows)
