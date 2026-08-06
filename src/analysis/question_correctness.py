"""
Case-study supplement: per-question correctness across all 6
(model x prompt) conditions. Two figures: Easy+Medium combined, and Hard on
its own -- Easy (2 of 40 questions have any wrong answer) and Medium (1 of
40) are each too sparse to justify a standalone chart, while Hard (38 of
51) is substantial enough to stand alone.

The heatmap itself is filtered to only questions with at least one wrong
answer among the 6 conditions -- an all-green row adds no information once
the accuracy summary box (see below) already states the overall percentage,
and dropping it keeps the chart focused on the actually-interesting cases.
The accuracy summary (Model x Prompt, % per tier) is computed from the
FULL, unfiltered tier(s) -- it must not be skewed by the filtering applied
to the chart rows above it.

Also produces a Hard-tier "hardest questions" ranking (by how many of the
6 conditions got it wrong) -- useful for pulling 2-3 concrete worked
examples into the Application/Case Study chapter, rather than only
reporting aggregate percentages.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.config import FIGURES_DIR, TABLES_DIR, MODEL_NAMES, PROMPT_TYPES, DIFFICULTIES
from src.utils.io import load_results

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


def correctness_grid(difficulties: list) -> pd.DataFrame:

    df = load_results()
    subset = df[df["Difficulty"].isin(difficulties)].copy()
    subset["Answer Correct"] = subset["Answer Correct"].astype(str).str.lower() == "true"
    subset["Difficulty"] = pd.Categorical(subset["Difficulty"], categories=DIFFICULTIES, ordered=True)

    subset = subset.sort_values(["Difficulty", "Category", "Question ID"])

    pivot = subset.pivot_table(
        index=["Difficulty", "Question ID"],
        columns=["Model", "Prompt"],
        values="Answer Correct",
        aggfunc="first",
    )

    columns = [(m, p) for m in MODEL_NAMES for p in PROMPT_TYPES]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(columns))

    return pivot


def accuracy_summary(difficulty: str) -> pd.DataFrame:
    """Model x Prompt accuracy (%) for one FULL tier -- independent of
    whatever row filtering the heatmap itself applies."""

    df = load_results()
    subset = df[df["Difficulty"] == difficulty].copy()
    subset["Answer Correct"] = subset["Answer Correct"].astype(str).str.lower() == "true"

    table = subset.groupby(["Model", "Prompt"])["Answer Correct"].mean().unstack("Prompt") * 100
    table = table.reindex(index=MODEL_NAMES, columns=PROMPT_TYPES)

    return table


def _accuracy_box_text(difficulties: list) -> str:
    """Stacks one Model x Prompt accuracy table per tier in the same box,
    each clearly labelled, rather than collapsing tiers together."""

    lines = ["Accuracy (%)"]
    for difficulty in difficulties:
        accuracy = accuracy_summary(difficulty)
        lines += ["", difficulty, f"{'Model':<8}{'CoT':>7}{'ToT':>7}", "-" * 22]
        for model in MODEL_NAMES:
            cot_pct = accuracy.loc[model, "cot"]
            tot_pct = accuracy.loc[model, "tot"]
            lines.append(f"{model.capitalize():<8}{cot_pct:>6.1f}%{tot_pct:>6.1f}%")

    return "\n".join(lines)


def plot_correctness_heatmap(difficulties: list, label: str, filename: str) -> None:

    full_grid = correctness_grid(difficulties)

    # Only questions with at least one wrong answer among the 6 conditions.
    wrong_mask = ~full_grid.astype(bool).all(axis=1)
    grid = full_grid[wrong_mask]

    n_rows, n_cols = grid.shape

    fig = plt.figure(figsize=(10.5, max(6, n_rows * 0.3)))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.2], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1])
    ax_box.axis("off")

    if n_rows == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No incorrect answers in this tier", ha="center", va="center", fontsize=11)
    else:
        values = grid.values.astype(float)  # True/False -> 1.0/0.0
        cmap = plt.matplotlib.colors.ListedColormap([CRITICAL, GOOD])
        ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        col_labels = [f"{m.capitalize()}\n{p.upper()}" for m, p in grid.columns]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        row_labels = [
            f"{qid} ({difficulty})" if len(difficulties) > 1 else qid
            for difficulty, qid in grid.index
        ]
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=7)

        # checkmark/cross on every cell -- correctness must not be colour-alone.
        for i in range(n_rows):
            for j in range(n_cols):
                mark = "✓" if values[i, j] == 1 else "✗"
                ax.text(j, i, mark, ha="center", va="center", color="white", fontsize=7)

        # Separators between models (every 2 columns = one model's CoT/ToT pair)
        for boundary in range(len(PROMPT_TYPES), n_cols, len(PROMPT_TYPES)):
            ax.axvline(boundary - 0.5, color="white", linewidth=3)

        # Separators between difficulty tiers (when combined)
        tier_sizes = grid.index.get_level_values("Difficulty").value_counts().reindex(
            [d for d in DIFFICULTIES if d in difficulties]
        )
        boundary = 0
        for size in tier_sizes[:-1]:
            boundary += size
            ax.axhline(boundary - 0.5, color="black", linewidth=1.5)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", length=0)

    # Accuracy summary box, separate from the heatmap, one table per tier,
    # computed on the full (unfiltered) tier(s).
    box_text = _accuracy_box_text(difficulties)
    ax_box.text(
        0.05, 0.95, box_text, transform=ax_box.transAxes, fontsize=9,
        fontfamily="monospace", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="black"),
    )

    fig.suptitle(
        f"{label} Correctness by Question and Condition (wrong-answer questions only)",
        fontsize=12, fontweight="bold",
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")


def hardest_questions_ranking(difficulty: str = "Hard") -> pd.DataFrame:

    grid = correctness_grid([difficulty])
    failures = (~grid.astype(bool)).sum(axis=1)

    ranking = failures.reset_index()
    ranking.columns = ["Difficulty", "Question ID", "Failures (of 6)"]
    ranking = ranking.drop(columns="Difficulty").sort_values("Failures (of 6)", ascending=False)

    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, f"{difficulty.lower()}_tier_hardest_questions.csv")
    ranking.to_csv(path, index=False)
    print(f"Saved -> {path}")

    return ranking


def run():

    plot_correctness_heatmap(["Easy", "Medium"], "Easy+Medium-Tier", "correctness_heatmap_easy_medium.png")
    plot_correctness_heatmap(["Hard"], "Hard-Tier", "correctness_heatmap_hard.png")

    ranking = hardest_questions_ranking("Hard")
    max_failures = ranking["Failures (of 6)"].max()
    print(f"\nUniversally hardest ({max_failures}/6 conditions wrong):")
    print(ranking[ranking["Failures (of 6)"] == max_failures])


if __name__ == "__main__":
    run()
