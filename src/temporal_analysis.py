import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

### CONSTANTS ###

SENTIMENT_COLORS = {
    "positive": "#59A14F",  
    "negative": "#D82923E1",  
    "neutral":  "#EB8825",
}
SIGNIFICANCE_COLOR = [
   "#C4AD66", "#6ACC65", "#D65F5F", "#B47CC7",
    "#4878CF", "#77BEDB", "#F28E2B", "#E15759",
    "#76B7B2", "#59A14F",
    ]

SENTIMENTS = ["positive", "negative", "neutral"]

### AGGREGATE FUNCTIONS ###

def compute_review_level(df: pd.DataFrame,
                          aspect_col = "aspect",
                          label_col = "final_pred",
                          review_id_col = "review_id",
                          year_col = "year") -> pd.DataFrame:
    """
    Aggregate chunk-level predictions to review level.

    1. For each (review_id, year, aspect) group, the dominant sentiment is the most frequently predicted polarity among all chunks where aspect is present.
    2. If no chunk detects the aspect as present, the review-level label is not_mentioned.
    """

    present = df[df[label_col] != "not_mentioned"].copy()

    def dominant(series):
        counts = series.value_counts()
        return counts.idxmax()

    review_sent = (
        present
        .groupby([review_id_col, year_col, aspect_col])[label_col]
        .apply(dominant)
        .reset_index()
        .rename(columns={label_col: "review_label"})
    )

    # All (review, aspect) pairs include not_mentioned rows
    all_pairs = (
        df[[review_id_col, year_col, aspect_col]]
        .drop_duplicates()
    )

    review_df = all_pairs.merge(review_sent, on=[review_id_col, year_col, aspect_col], how="left")
    review_df["review_label"] = review_df["review_label"].fillna("not_mentioned")
    return review_df


def compute_aspect_significance(review_df: pd.DataFrame,
                                 aspect_col = "aspect",
                                 year_col = "year",
                                 review_id_col = "review_id") -> pd.DataFrame:
    """
    Calculate the aspect significance following the formula:
        ψ_k(year) = |R_k(year)| / |R(year)|

    where R_k(year) = unique reviews in that year where aspect k was detected in at least one chunk, and R(year) = all unique reviews in that year.
    """

    total_per_year = (
        review_df.groupby(year_col)[review_id_col]
        .nunique()
        .rename("total_reviews")
        .reset_index()
    )

    present_per_year = (
        review_df[review_df["review_label"] != "not_mentioned"]
        .groupby([year_col, aspect_col])[review_id_col]
        .nunique()
        .rename("n_present")
        .reset_index()
    )

    sig = present_per_year.merge(total_per_year, on=year_col)
    sig["psi"] = sig["n_present"] / sig["total_reviews"]
    return sig


def compute_sentiment_distribution(review_df: pd.DataFrame,
                                    aspect_col = "aspect",
                                    year_col = "year",
                                    review_id_col = "review_id") -> pd.DataFrame:
    """
    Calculate the sentiment distribution for each aspect and year following the formula:
        ξ_k^ρ(year) = |R_k^ρ(year)| / |R_k(year)|

    where R_k^ρ(year) is the subset of reviews where the dominant sentiment toward aspect k is polarity ρ, and R_k(year) is the set of reviews where aspect k is present (regardless of sentiment).
    """

    present = review_df[review_df["review_label"] != "not_mentioned"].copy()

    if present.empty:
        return pd.DataFrame()

    aspect_total = (
        present
        .groupby([year_col, aspect_col])[review_id_col]
        .nunique()
        .rename("n_aspect_present")
        .reset_index()
    )

    aspect_sent = (
        present
        .groupby([year_col, aspect_col, "review_label"])[review_id_col]
        .nunique()
        .rename("n_polarity")
        .reset_index()
    )

    dist = aspect_sent.merge(aspect_total, on=[year_col, aspect_col])
    dist["xi"] = dist["n_polarity"] / dist["n_aspect_present"]
    dist["n_reviews"] = dist["n_polarity"]   

    dist = dist.rename(columns={"review_label": "sentiment"})

    return dist

### SUMMARY TABLE ###

def build_summary_table(sig_df: pd.DataFrame,
                         dist_df: pd.DataFrame,
                         aspect_col = "aspect",
                         year_col = "year") -> pd.DataFrame:

    pivot = dist_df.pivot_table(
        index=[year_col, aspect_col],
        columns="sentiment",
        values="xi",
        aggfunc="first",
    ).reset_index()

    # Ensure all sentiment columns exist
    for s in SENTIMENTS:
        if s not in pivot.columns:
            pivot[s] = np.nan

    pivot = pivot.rename(columns={s: f"xi_{s}" for s in SENTIMENTS})
    summary = sig_df[[year_col, aspect_col, "psi", "n_present", "total_reviews"]].merge(
        pivot, on=[year_col, aspect_col], how="left"
    )

    sent_cols = [f"xi_{s}" for s in SENTIMENTS]
    summary["dominant_sentiment"] = summary[sent_cols].idxmax(axis=1).str.replace("xi_", "")
    return summary.sort_values([aspect_col, year_col]).reset_index(drop=True)


### PLOTS ###

def _savefig(fig, path: Path, fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(f".{fmt}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {path.with_suffix(f'.{fmt}')}")

def plot_aspect_significance_heatmap(sig_df: pd.DataFrame,
                                      aspect_col: str,
                                      year_col: str,
                                      top_n: int,
                                      output_dir: Path,
                                      fmt: str):
    """Heatmap for aspect significance per year for each aspects."""

    avg_sig = (
        sig_df.groupby(aspect_col)["psi"]
        .mean()
        .index.tolist()
    )

    pivot = (
        sig_df[sig_df[aspect_col].isin(avg_sig)]
        .pivot(index=aspect_col, columns=year_col, values="psi")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.9),
                                    max(4, len(avg_sig) * 0.6)))

    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Proportion of revies (%)"},
        ax=ax,
    )
    ax.set_title(f"Aspect Significance by Aspect and Year", fontsize=13, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Aspect")
    plt.tight_layout()
    _savefig(fig, output_dir / "significance_heatmap", fmt)

def plot_overall_significance_bar(sig_df, dist_df, aspect_col, year_col,
                                   top_n, output_dir, fmt, aspect_color_map=None):
    """Bar plot for overall aspect significance (ψ) and sentiment distribution (ξ) across all years."""

    avg_psi = (
        sig_df.groupby(aspect_col)["n_present"]
        .sum()
        .sort_values(ascending=True)
    )
    total_reviews = avg_psi.sum()

    aspect_colors = [aspect_color_map[a] for a in avg_psi.index] 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(avg_psi) * 0.55)))

    # Left panel: Aspect significance
    bars = ax1.barh(avg_psi.index, avg_psi.values,
                    color=aspect_colors, edgecolor="white", linewidth=0.5)
    
    labels = [
        f"{count:.0f} ({count/total_reviews:.1%})"
        for count in avg_psi.values
    ]

    ax1.bar_label(
        bars,
        labels=labels,
        padding=3,
        fontsize=8,
    )
    ax1.set_xlabel("Total Number of Reviews")
    ax1.set_title("Aspect Significance")
    ax1.set_axisbelow(True)
    ax1.grid(axis="x", color="lightgrey", linewidth=0.8)
    ax1.set_facecolor("white")

    # Right panel: Sentiment distribution
    overall_sentiment = (
        dist_df.groupby("sentiment")["n_reviews"]
        .sum()
        .reindex(SENTIMENTS, fill_value=0)
        .sort_values(ascending=True)
    )

    total_sentiments = overall_sentiment.sum()

    bar_colors = [SENTIMENT_COLORS[s] for s in overall_sentiment.index]
    bars2 = ax2.barh(
        [s.capitalize() for s in overall_sentiment.index],
        overall_sentiment.values,
        color=bar_colors,
        edgecolor="white", linewidth=0.5, height=0.5,
    )

    labels = [
        f"{count:.0f} ({count/total_sentiments:.1%})"
        for count in overall_sentiment.values
    ]

    ax2.bar_label(
        bars2,
        labels=labels,
        padding=3,
        fontsize=8,
    )

    ax2.set_xlabel("Total Number of Reviews")
    ax2.set_title("Overall Sentiment Distribution")
    ax2.set_xlim(0, overall_sentiment.values.max() * 1.2)
    ax2.set_axisbelow(True)
    ax2.grid(axis="x", color="lightgrey", linewidth=0.8)
    ax2.set_facecolor("white")

    fig.suptitle(f"Overall Aspect Overview", fontsize=13)
    plt.tight_layout()
    _savefig(fig, output_dir / "significance_overall_bar", fmt)

def plot_negative_ratio_heatmap(dist_df, aspect_col, year_col, output_dir, fmt):
    """Heatmap of negative sentiment ratio (xi_negative) per aspect per year."""

    pivot = (
        dist_df[dist_df["sentiment"] == "negative"]
        .pivot(index=aspect_col, columns=year_col, values="xi")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.9),
                                    max(4, len(pivot.index) * 0.6)))

    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Negative sentiment ratio"},
        vmin=0, vmax=pivot.values.max(),
        ax=ax,
    )
    ax.set_title("Negative Sentiment Ratio by Aspect and Year", fontsize=13, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Aspect")
    plt.tight_layout()
    _savefig(fig, output_dir / "negative_ratio_heatmap", fmt)

def plot_sentiment_stacked_bar_per_aspect(dist_df: pd.DataFrame,
                                           aspect_col: str,
                                           top_n: int,
                                           output_dir: Path,
                                           fmt: str):
    """Horizontal stacked bar chart to see proportion of sentiment for each aspect across all years."""

    agg = (
        dist_df.groupby([aspect_col, "sentiment"])["n_reviews"]
        .sum()
        .reset_index()
    )

    totals = agg.groupby(aspect_col)["n_reviews"].sum()

    sig_order = totals.nlargest(top_n).index.tolist()
    sig_order = sig_order[::-1]

    pivot = (
        agg.pivot(index=aspect_col, columns="sentiment", values="n_reviews")
        .reindex(index=sig_order, columns=SENTIMENTS)
        .fillna(0)
    )

    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig_height = max(4, len(sig_order) * 0.55)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    y_pos = np.arange(len(sig_order))
    left = np.zeros(len(sig_order))

    for sent in SENTIMENTS:
        counts = pivot[sent].values
        vals = pct[sent].values

        ax.barh(y_pos, vals, left=left, height=0.8,
                label=sent, color=SENTIMENT_COLORS[sent],
                edgecolor="white", linewidth=0.5)

        for i, (c, p, l) in enumerate(zip(counts, vals, left)):
            if c <= 0:
                continue
            label = f"{int(c)} ({p:.0f}%)"

            if sent == "positive":
                ax.text(l + p / 2, y_pos[i], label,
                         ha="center", va="center", fontsize=7.5,
                         color="black", fontweight="bold")

            elif sent == "negative":
                ax.text(l + p/2, y_pos[i], label,
                         ha="center", va="center", fontsize=7.5,
                         color="black", fontweight="bold")

            else:  
                ax.text(102, y_pos[i], label,
                         ha="left", va="center", fontsize=8,
                         color="black", fontweight="bold")

        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_order, fontsize=9)
    ax.set_xlabel("Percentage of Reviews (%)")
    ax.set_xlim(0, 110)  
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_xticks(range(0, 101, 20)) 
    
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color="lightgrey", linewidth=0.8, alpha=1.0)
    ax.set_facecolor("white")

    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=len(SENTIMENTS), frameon=False)

    ax.set_title("Sentiment Proportion by Aspect", fontsize=13, pad=10)

    plt.tight_layout()
    _savefig(fig, output_dir / "sentiment_stacked_bar", fmt)

#### MAIN FUNCTION ####

def aggregate_preds(prediction_df, output_dir, year_col="year", aspect_col="aspect", label_col="final_pred", review_id_col="review_id", chunk_level=False):
    """Main function to calculate aspect significance and sentiment distribution."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prediction_df.copy()
    print(f"Prediction rows: {len(df)}")

    df[year_col] = df[year_col].astype(int)

    review_df = compute_review_level(
        df,
        aspect_col    = aspect_col,
        label_col     = label_col,
        review_id_col = review_id_col,
        year_col      = year_col,
    )
    print(f"Review-level rows: {len(review_df):,}")

    # Calculate aspect significance (ψ) and sentiment distribution (ξ)
    sig_df = compute_aspect_significance(
        review_df,
        aspect_col    = aspect_col,
        year_col      = year_col,
        review_id_col = review_id_col,
    )

    dist_df = compute_sentiment_distribution(
        review_df,
        aspect_col    = aspect_col,
        year_col      = year_col,
        review_id_col = review_id_col,
    )

    sig_path  = output_dir / "aspect_significance.csv"
    dist_path = output_dir / "sentiment_distribution.csv"
    sig_df.to_csv(sig_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    print(f"\n  Saved aspect significance to {sig_path}")
    print(f"  Saved sentiment distribution to {dist_path}")

    # Summary table
    summary = build_summary_table(sig_df, dist_df,
                                    aspect_col=aspect_col,
                                    year_col=year_col)
    summary_path = output_dir / "temporal_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved temporal summary to {summary_path}")

    return sig_df, dist_df, summary

def run_analysis(sig_df, dist_df, output_dir, year_col="year", aspect_col="aspect", top_n=10, fmt="png"):
    """Main function to display summary statistics and generate visualizations for temporal analysis."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aspects = sorted(sig_df[aspect_col].unique())

    ASPECT_COLOR_MAP = {
        aspect: SIGNIFICANCE_COLOR[i % len(SIGNIFICANCE_COLOR)]
        for i, aspect in enumerate(aspects)
    }

    # Display summary statistics
    print(f"\n  ASPECT SIGNIFICANCE SUMMARY (average ψ across years)")
    avg_psi = (
        sig_df.groupby(aspect_col)["psi"]
        .mean()
        .sort_values(ascending=False)
    )
    for aspect, psi in avg_psi.items():
        print(f"  {aspect:<30} ψ = {psi:.4f}  ({psi*100:.2f}%)")

    print(f"\n  SENTIMENT DISTRIBUTION SUMMARY (average ξ across years)")
    avg_xi = (
        dist_df.groupby([aspect_col, "sentiment"])["xi"]
        .mean()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENTS, fill_value=0)
    )
    print(f"  {'Aspect':<30} {'Positive':>10} {'Negative':>10} {'Neutral':>10}  Dominant")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}  {'─'*10}")
    for aspect, row in avg_xi.iterrows():
        dominant = row.idxmax()
        print(f"  {aspect:<30} {row['positive']:>9.3f}  {row['negative']:>9.3f}"
                f"  {row['neutral']:>9.3f}  {dominant}")

    # Display plots
    top_n = min(top_n, len(aspects))
    print(f"\n Saving plots")

    plot_overall_significance_bar(sig_df, dist_df, aspect_col, year_col,
                                   top_n, output_dir, fmt, aspect_color_map=ASPECT_COLOR_MAP)

    plot_aspect_significance_heatmap(sig_df, aspect_col, year_col,
                                      top_n, output_dir, fmt)
    
    plot_negative_ratio_heatmap(dist_df, aspect_col, year_col, output_dir, fmt)

    plot_sentiment_stacked_bar_per_aspect(dist_df, aspect_col, top_n, output_dir, fmt)

    return None