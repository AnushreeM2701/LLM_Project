"""
Static counterpart to the dashboard's "Error Type Frequency Count" panel:
per-prompt common-wrong-question intersection (equal N across models).
Hover meanings/examples are embedded as a legend panel in the image.
"""

import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from config.config import FIGURES_DIR, TABLES_DIR, MODEL_NAMES, PROMPT_TYPES
from src.utils.io import load_results
from src.analysis.error_definitions import ERROR_TYPE_DEFINITIONS

def common_wrong_hard_questions(df: pd.DataFrame, prompt: str) -> set:
    """Hard-tier Question IDs all 3 models got wrong, within one prompt."""

    sub = df[(df["Difficulty"] == "Hard") & (df["Prompt"] == prompt)]
    wrong_sets = [set(sub[(sub["Model"] == m) & (~sub["Answer Correct"])]["Question ID"]) for m in MODEL_NAMES]
    return set.intersection(*wrong_sets)


def render_legend_panel(ax, error_types: list, examples: dict) -> None:
    """Meaning + example per error type, in place of hover tooltips."""

    ax.axis("off")
    ax.set_title("What each error type means (hover content, shown here instead)", fontsize=10.5, fontweight="bold", loc="left")
    lines = []
    for t in error_types:
        meaning = ERROR_TYPE_DEFINITIONS.get(t, "No definition available.")
        example = examples.get(t)
        line = f"{t}  -  {meaning}"
        if example:
            line += f'\n    Example (this project): "{example}"'
        lines.append(line)
    ax.text(0.0, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=8.5, ha="left", va="top")


def run():

    df = load_results()
    df = df.copy()
    df["Answer Correct"] = df["Answer Correct"].astype(str).str.lower() == "true"

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for prompt in PROMPT_TYPES:

        common = common_wrong_hard_questions(df, prompt)
        pool = df[(df["Difficulty"] == "Hard") & (df["Prompt"] == prompt) & (df["Question ID"].isin(common)) & (~df["Answer Correct"])]

        per_model_counts = {}
        shared_max = 0
        all_examples = {}
        combined_freq = []
        for model in MODEL_NAMES:
            model_wrong = pool[pool["Model"] == model]
            counts = Counter(model_wrong["Error Type"])
            freq = pd.DataFrame(sorted(counts.items(), key=lambda kv: kv[1]), columns=["Error Type", "Count"])
            combined_freq.append(freq.assign(Model=model))
            per_model_counts[model] = {"freq": freq, "n": len(model_wrong)}
            if len(freq):
                shared_max = max(shared_max, freq["Count"].max())
            for _, row in model_wrong.iterrows():
                t = row["Error Type"]
                if t not in all_examples and str(row["Error Subtype"]).strip():
                    all_examples[t] = row["Error Subtype"]

        combined_path = os.path.join(TABLES_DIR, f"error_type_frequency_hard_{prompt}.csv")
        pd.concat(combined_freq, ignore_index=True)[["Model", "Error Type", "Count"]].to_csv(
            combined_path, index=False
        )
        print(f"Saved -> {combined_path}")

        xlim = (0, math.ceil(shared_max * 1.15) or 1)
        all_error_types = sorted(
            {t for model in MODEL_NAMES for t in per_model_counts[model]["freq"]["Error Type"]},
            key=lambda t: -sum(per_model_counts[m]["freq"].set_index("Error Type")["Count"].get(t, 0) for m in MODEL_NAMES),
        )

        fig = plt.figure(figsize=(18, 9.5))
        gs = fig.add_gridspec(2, len(MODEL_NAMES), height_ratios=[3, 1.7], hspace=0.4, wspace=0.6)

        for col, model in enumerate(MODEL_NAMES):
            freq = per_model_counts[model]["freq"]
            n = per_model_counts[model]["n"]
            ax = fig.add_subplot(gs[0, col])
            bars = ax.barh(freq["Error Type"], freq["Count"], color="#2a78d6", edgecolor="black", linewidth=0.5)
            for bar, count in zip(bars, freq["Count"]):
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                         str(count), va="center", fontsize=8)
            ax.set_title(f"{model.capitalize()} (n={n} errors)", fontsize=10.5, fontweight="bold")
            ax.set_xlabel("Occurrences")
            ax.set_xlim(xlim)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        legend_ax = fig.add_subplot(gs[1, :])
        render_legend_panel(legend_ax, all_error_types, all_examples)

        fig.suptitle(
            f"Error Type Frequency by Model - Hard Tier Only ({prompt.upper()}) - "
            f"restricted to Hard-tier questions that ALL 3 models got wrong (n={len(common)})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        path = os.path.join(FIGURES_DIR, f"error_type_frequency_hard_by_model_{prompt}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {path}")


if __name__ == "__main__":
    run()
