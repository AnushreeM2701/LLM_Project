"""
RQ1 supplement / dashboard "Error Heatmap" counterpart: Model x Difficulty x
Error Type (top 6), balanced 40/40/40 pool, CoT vs ToT on a shared scale.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from config.config import FIGURES_DIR, MODEL_NAMES, DIFFICULTIES, PROMPT_TYPES
from src.utils.io import load_results
from src.analysis.question_execution_time import balanced_question_pool

TOP_N = 6

# Sequential single-hue (blue) ramp, light -> dark.
CMAP = plt.cm.Blues


def top_error_types(n: int = TOP_N) -> list:

    df = load_results()
    errors = df[df["Answer Correct"].astype(str).str.lower() == "false"]

    return errors["Error Type"].value_counts().head(n).index.tolist()


def model_difficulty_by_category_grid(prompt: str, categories: list, df: pd.DataFrame, pool: dict) -> pd.DataFrame:

    errors = df[
        (df["Answer Correct"].astype(str).str.lower() == "false")
        & (df["Prompt"] == prompt)
        & (df["Error Type"].isin(categories))
        & df.apply(lambda r: r["Question ID"] in pool[r["Difficulty"]], axis=1)
    ]

    row_index = pd.MultiIndex.from_product(
        [MODEL_NAMES, DIFFICULTIES], names=["Model", "Difficulty"]
    )

    grid = errors.groupby(["Model", "Difficulty", "Error Type"]).size().unstack(fill_value=0)
    grid = grid.reindex(index=row_index, columns=categories, fill_value=0)

    return grid


def plot_error_heatmaps() -> None:

    df = load_results()
    pool = balanced_question_pool(df)

    categories = top_error_types()
    grids = {p: model_difficulty_by_category_grid(p, categories, df, pool) for p in PROMPT_TYPES}

    vmax = max(g.values.max() for g in grids.values())
    n_rows = len(MODEL_NAMES) * len(DIFFICULTIES)
    n_diff = len(DIFFICULTIES)

    fig, axes = plt.subplots(1, len(PROMPT_TYPES), figsize=(15, 8), sharey=False)

    row_labels = DIFFICULTIES * len(MODEL_NAMES)

    for ax, prompt in zip(axes, PROMPT_TYPES):

        grid = grids[prompt]
        im = ax.imshow(grid.values, cmap=CMAP, vmin=0, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(prompt.upper(), fontsize=11, fontweight="bold")

        # Model name centred once per its 3-row difficulty group.
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for group_idx, model in enumerate(MODEL_NAMES):
            center_row = group_idx * n_diff + (n_diff - 1) / 2
            ax.text(-0.30, center_row, model.capitalize(), transform=trans,
                     fontweight="bold", fontsize=9, ha="left", va="center")

        # Zeros shown explicitly -- "no errors observed" is informative here.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid.values[i, j]
                if value == 0:
                    text_color = "#999999"
                elif value > vmax * 0.6:
                    text_color = "white"
                else:
                    text_color = "black"
                ax.text(j, i, str(value), ha="center", va="center",
                         color=text_color, fontsize=8)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", length=0)

        # Thicker separator every 3 rows (one model's Easy/Medium/Hard group).
        for boundary in range(len(DIFFICULTIES), n_rows, len(DIFFICULTIES)):
            ax.axhline(boundary - 0.5, color="white", linewidth=3)

    fig.colorbar(im, ax=axes, shrink=0.7, label="Error count", pad=0.02)
    fig.suptitle(
        f"RQ1 - Error Heatmap (Model x Difficulty x Error Type) - "
        f"40 Easy + 40 Medium + 40 Hard = 120 questions",
        fontsize=13, fontweight="bold")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "error_heatmap_top6_by_prompt.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")
    for prompt, grid in grids.items():
        print(f"\n--- {prompt} ---")
        print(grid)


def run():
    plot_error_heatmaps()


if __name__ == "__main__":
    run()
