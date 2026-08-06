"""
Reasoning length (Step Count) vs. correctness -- a different angle from RQ2
(which looks at WHERE within existing steps an error occurs, not how many
steps a response used in total). Question asked here: do incorrect
responses tend to be longer (overthinking/rambling) or shorter
(insufficient reasoning) than correct ones?

Broken down three ways per the requested granularity:
- Model-wise and level-wise: summary table (Model x Difficulty x
  Correct/Incorrect -> N, mean, median step count).
- Question-wise: per-question scatter, one panel per model, points
  coloured by correctness (green/red, matching the same status-colour
  convention as src/analysis/question_correctness.py).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from config.config import FIGURES_DIR, TABLES_DIR, MODEL_NAMES, PROMPT_TYPES, DIFFICULTIES
from src.utils.io import load_results

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


def step_count_summary() -> pd.DataFrame:
    """Model x Difficulty x Correct/Incorrect -> N, mean, median step count."""

    df = load_results()
    df = df.copy()
    df["Answer Correct"] = df["Answer Correct"].astype(str).str.lower() == "true"
    df["Step Count"] = pd.to_numeric(df["Step Count"], errors="coerce")

    rows = []
    for model in MODEL_NAMES:
        for difficulty in DIFFICULTIES:
            for correct in [True, False]:
                subset = df[
                    (df["Model"] == model)
                    & (df["Difficulty"] == difficulty)
                    & (df["Answer Correct"] == correct)
                ]
                if len(subset) == 0:
                    continue
                rows.append({
                    "Model": model,
                    "Difficulty": difficulty,
                    "Correct": correct,
                    "N": len(subset),
                    "Mean Step Count": subset["Step Count"].mean(),
                    "Median Step Count": subset["Step Count"].median(),
                })

    return pd.DataFrame(rows)


def plot_step_count_by_question(prompt: str) -> None:

    df = load_results()
    subset = df[df["Prompt"] == prompt].copy()
    subset["Answer Correct"] = subset["Answer Correct"].astype(str).str.lower() == "true"
    subset["Step Count"] = pd.to_numeric(subset["Step Count"], errors="coerce")

    # Difficulty is a secondary dimension here, so it gets its own visual
    # channel (marker shape) rather than overloading colour, which is
    # already carrying correctness.
    difficulty_markers = {"Easy": "o", "Medium": "s", "Hard": "^"}

    fig, axes = plt.subplots(len(MODEL_NAMES), 1, figsize=(13, 3.2 * len(MODEL_NAMES)), sharex=True)

    for ax, model in zip(axes, MODEL_NAMES):

        model_df = subset[subset["Model"] == model]

        for difficulty, marker in difficulty_markers.items():
            for correct, color, status in [(True, GOOD, "Correct"), (False, CRITICAL, "Incorrect")]:
                group = model_df[
                    (model_df["Difficulty"] == difficulty) & (model_df["Answer Correct"] == correct)
                ]
                if len(group) == 0:
                    continue
                ax.scatter(
                    group["Question Number"], group["Step Count"],
                    label=f"{difficulty} - {status}", color=color, marker=marker,
                    s=45, alpha=0.8, edgecolor="black", linewidth=0.4,
                )

        ax.set_title(model.capitalize(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Step Count")
        ax.grid(linestyle="--", alpha=0.3)

    # De-duplicated legend: 3 shapes (difficulty) x 2 colours (correctness),
    # built once rather than repeating per-model auto-legends.
    from matplotlib.lines import Line2D
    shape_handles = [
        Line2D([0], [0], marker=m, color="black", linestyle="", markerfacecolor="gray",
               markersize=8, label=d)
        for d, m in difficulty_markers.items()
    ]
    color_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="", markerfacecolor=c,
               markersize=8, label=label)
        for c, label in [(GOOD, "Correct"), (CRITICAL, "Incorrect")]
    ]
    axes[0].legend(handles=shape_handles + color_handles, loc="upper left", ncol=5, fontsize=8)
    axes[-1].set_xlabel("Question Number (1-131)")

    fig.suptitle(f"Step Count by Question, per Model (shape=Difficulty, colour=Correctness) ({prompt.upper()})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"step_count_by_question_{prompt}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")


def run():

    os.makedirs(TABLES_DIR, exist_ok=True)
    summary = step_count_summary()
    path = os.path.join(TABLES_DIR, "step_count_summary.csv")
    summary.to_csv(path, index=False)
    print(f"Saved -> {path}")
    print(summary)

    for prompt in PROMPT_TYPES:
        plot_step_count_by_question(prompt)


if __name__ == "__main__":
    run()
