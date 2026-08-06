"""
Case-study supplement: per-question execution time, one panel per model
(small multiples), points coloured by difficulty tier. Replaces an earlier
heatmap version -- a single outlier (NT_H_004 on Mistral/ToT, 894s) blew out
the heatmap's shared colour scale and washed out every other cell. A
scatter plot doesn't have that problem: each point stands on its own axis
position regardless of how extreme its neighbours are.

One figure per prompt type (CoT/ToT), since combining both on one axis
would need a second visual encoding (shape) on top of the colour already
used for difficulty.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from config.config import FIGURES_DIR, MODEL_NAMES, PROMPT_TYPES, DIFFICULTIES
from src.utils.io import load_results

# Difficulty reads as a severity/status signal here (easy=safe, hard=costly),
# so it borrows the status palette rather than the categorical one.
DIFFICULTY_COLORS = {"Easy": "#0ca30c", "Medium": "#fab219", "Hard": "#d03b3b"}

# Fixed categorical order, matching the same 3 hues used elsewhere in this
# project's figures (never cycled/reassigned per chart).
MODEL_COLORS = {"gemini": "#2a78d6", "groq": "#eb6834", "mistral": "#1baf7a"}


def plot_execution_time_by_question(prompt: str) -> None:

    df = load_results()
    subset = df[df["Prompt"] == prompt].copy()
    subset["Execution Time (s)"] = pd.to_numeric(subset["Execution Time (s)"], errors="coerce")

    fig, axes = plt.subplots(len(MODEL_NAMES), 1, figsize=(13, 3.2 * len(MODEL_NAMES)), sharex=True)

    for ax, model in zip(axes, MODEL_NAMES):

        model_df = subset[subset["Model"] == model]

        for difficulty in DIFFICULTIES:
            diff_df = model_df[model_df["Difficulty"] == difficulty]
            ax.scatter(
                diff_df["Question Number"], diff_df["Execution Time (s)"],
                label=difficulty, color=DIFFICULTY_COLORS[difficulty],
                s=45, alpha=0.85, edgecolor="black", linewidth=0.4,
            )

        ax.set_title(model.capitalize(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Execution Time (s)")
        ax.grid(linestyle="--", alpha=0.3)

    axes[0].legend(title="Difficulty", loc="upper left", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Question Number (1-131)")

    fig.suptitle(f"Execution Time by Question, per Model ({prompt.upper()})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"execution_time_by_question_{prompt}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")


def plot_execution_time_by_difficulty(prompt: str) -> None:
    """Transposed view of plot_execution_time_by_question: one panel per
    DIFFICULTY tier, points coloured by MODEL, x-axis is the question's
    position within its own tier (1..N) rather than the global 1-131
    numbering -- so Easy/Medium (40 questions each) and Hard (51) all
    start at 1, matching how the tiers are naturally compared."""

    df = load_results()
    subset = df[df["Prompt"] == prompt].copy()
    subset["Execution Time (s)"] = pd.to_numeric(subset["Execution Time (s)"], errors="coerce")

    fig, axes = plt.subplots(len(DIFFICULTIES), 1, figsize=(13, 3.2 * len(DIFFICULTIES)), sharex=False)

    for ax, difficulty in zip(axes, DIFFICULTIES):

        diff_df = subset[subset["Difficulty"] == difficulty].copy()

        # Position within this tier, not the global Question Number.
        tier_question_ids = sorted(diff_df["Question ID"].unique())
        position = {qid: i + 1 for i, qid in enumerate(tier_question_ids)}
        diff_df["Tier Position"] = diff_df["Question ID"].map(position)

        for model in MODEL_NAMES:
            model_df = diff_df[diff_df["Model"] == model]
            ax.scatter(
                model_df["Tier Position"], model_df["Execution Time (s)"],
                label=model.capitalize(), color=MODEL_COLORS[model],
                s=45, alpha=0.85, edgecolor="black", linewidth=0.4,
            )

        ax.set_title(difficulty, fontsize=11, fontweight="bold")
        ax.set_ylabel("Execution Time (s)")
        ax.set_xlabel(f"Question Number (1-{len(tier_question_ids)})")
        ax.grid(linestyle="--", alpha=0.3)

    axes[0].legend(title="Model", loc="upper left", ncol=3, fontsize=8)

    fig.suptitle(f"Execution Time by Question, per Difficulty ({prompt.upper()})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"execution_time_by_difficulty_{prompt}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")


def run():
    for prompt in PROMPT_TYPES:
        plot_execution_time_by_question(prompt)
        plot_execution_time_by_difficulty(prompt)


if __name__ == "__main__":
    run()
