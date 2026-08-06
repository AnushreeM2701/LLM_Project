"""
RQ1 supplement: Model x Difficulty x Error Type heatmap, for the top 6 most
common error categories, plotted separately for CoT and ToT on a shared
colour scale so the two are visually comparable.

Note on sparsity: with 136 total errors spread across 3 models x 3
difficulty tiers x 6 categories x 2 prompts, only 38/108 cells are
populated (many legitimate zeros). Zeros are shown explicitly rather than
left blank, since "no errors of this type observed" is itself informative
here, not missing data -- but this sparsity is worth naming as a
limitation on small-N subgroup comparisons if you discuss this figure.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from config.config import FIGURES_DIR, MODEL_NAMES, DIFFICULTIES, PROMPT_TYPES
from src.utils.io import load_results

TOP_N = 6

# Sequential single-hue (blue) ramp, light -> dark.
CMAP = plt.cm.Blues


def top_error_types(n: int = TOP_N) -> list:

    df = load_results()
    errors = df[df["Answer Correct"].astype(str).str.lower() == "false"]

    return errors["Error Type"].value_counts().head(n).index.tolist()


def model_difficulty_by_category_grid(prompt: str, categories: list) -> pd.DataFrame:

    df = load_results()
    errors = df[
        (df["Answer Correct"].astype(str).str.lower() == "false")
        & (df["Prompt"] == prompt)
        & (df["Error Type"].isin(categories))
    ]

    row_index = pd.MultiIndex.from_product(
        [MODEL_NAMES, DIFFICULTIES], names=["Model", "Difficulty"]
    )

    grid = errors.groupby(["Model", "Difficulty", "Error Type"]).size().unstack(fill_value=0)
    grid = grid.reindex(index=row_index, columns=categories, fill_value=0)

    return grid


def plot_error_heatmaps() -> None:

    categories = top_error_types()
    grids = {p: model_difficulty_by_category_grid(p, categories) for p in PROMPT_TYPES}

    vmax = max(g.values.max() for g in grids.values())
    n_rows = len(MODEL_NAMES) * len(DIFFICULTIES)
    n_diff = len(DIFFICULTIES)

    fig, axes = plt.subplots(1, len(PROMPT_TYPES), figsize=(15, 8), sharey=False)

    # Row labels show only the difficulty; the model name is written once
    # per group (see the blended-transform ax.text block below), matching
    # the grouped-header style of the reference figure this was modelled on
    # rather than repeating "Gemini" on every one of its three rows.
    row_labels = DIFFICULTIES * len(MODEL_NAMES)

    for ax, prompt in zip(axes, PROMPT_TYPES):

        grid = grids[prompt]
        im = ax.imshow(grid.values, cmap=CMAP, vmin=0, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(prompt.upper(), fontsize=11, fontweight="bold")

        # Model name written once, bold, vertically centred on its group of
        # three difficulty rows, positioned left of the difficulty labels
        # via a transform blending axes-fraction x with data-coordinate y.
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for group_idx, model in enumerate(MODEL_NAMES):
            center_row = group_idx * n_diff + (n_diff - 1) / 2
            ax.text(-0.30, center_row, model.capitalize(), transform=trans,
                     fontweight="bold", fontsize=9, ha="left", va="center")

        # Direct labels on every cell, including explicit zeros -- see
        # module docstring on why zeros are shown rather than left blank.
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

        # Thicker separators between models (every 3 rows = one model's
        # Easy/Medium/Hard group) so the Model grouping reads clearly.
        for boundary in range(len(DIFFICULTIES), n_rows, len(DIFFICULTIES)):
            ax.axhline(boundary - 0.5, color="white", linewidth=3)

    fig.colorbar(im, ax=axes, shrink=0.7, label="Error count", pad=0.02)
    fig.suptitle(f"Top {TOP_N} Error Types by Model and Difficulty (CoT vs. ToT)",
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
