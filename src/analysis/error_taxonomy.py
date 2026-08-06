"""
RQ1: What are the most common types of mathematical reasoning errors?

Descriptive error-type distribution plus a chi-square (or Fisher's exact,
for small tables) test of independence on Model x Error Type, so "error
profiles differ by model" is a tested claim, not just an eyeballed table.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from config.config import FIGURES_DIR, TABLES_DIR, MODEL_NAMES
from src.utils.io import load_results
from src.utils.stats import independence_test


def error_distribution() -> pd.DataFrame:

    df = load_results()
    errors = df[df["Error Type"].astype(str).str.strip() != "Correct"]

    counts = errors["Error Type"].value_counts().reset_index()
    counts.columns = ["Error Type", "Count"]

    return counts


def error_type_by_model_contingency() -> pd.DataFrame:

    df = load_results()
    errors = df[df["Error Type"].astype(str).str.strip() != "Correct"]

    table = pd.crosstab(errors["Model"], errors["Error Type"])
    table = table.reindex(index=[m for m in MODEL_NAMES if m in table.index])

    return table


def plot_error_distribution(dist: pd.DataFrame) -> None:
    """Headline overall ranking (all models/prompts combined) -- a single
    ordered series, so one flat colour throughout is correct (this isn't a
    categorical-identity chart, just a magnitude ranking)."""

    ordered = dist.sort_values("Count", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(ordered["Error Type"], ordered["Count"], color="#2a78d6", edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, ordered["Count"]):
        ax.text(bar.get_width() + max(ordered["Count"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                 str(count), va="center", fontsize=9)

    ax.set_xlabel("Number of Errors")
    ax.set_title("Distribution of Reasoning Error Types (all models, all prompts)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "error_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")


def run():

    os.makedirs(TABLES_DIR, exist_ok=True)

    dist = error_distribution()
    dist_path = os.path.join(TABLES_DIR, "rq1_error_distribution.csv")
    dist.to_csv(dist_path, index=False)
    print(f"Saved -> {dist_path}")
    print(dist)
    print()

    plot_error_distribution(dist)

    contingency = error_type_by_model_contingency()
    contingency_path = os.path.join(TABLES_DIR, "rq1_error_type_by_model.csv")
    contingency.to_csv(contingency_path)
    print(f"Saved -> {contingency_path}")
    print(contingency)
    print()

    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        test = independence_test(contingency)
        test_df = pd.DataFrame([test])
        test_path = os.path.join(TABLES_DIR, "rq1_independence_test.csv")
        test_df.to_csv(test_path, index=False)
        print(f"Saved -> {test_path}")
        print(test)
    else:
        print("Not enough model/error-type variety yet for an independence test.")

    return dist, contingency


if __name__ == "__main__":
    run()
