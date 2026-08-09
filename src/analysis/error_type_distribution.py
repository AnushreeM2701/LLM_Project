"""
Static counterpart to the dashboard's "Error Type Distribution" panel:
Error Type x Difficulty tables per model/prompt, balanced 40/40/40 pool.
Glossary is embedded in the same image (not a separate file).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from config.config import FIGURES_DIR, TABLES_DIR, MODEL_NAMES, PROMPT_TYPES, DIFFICULTIES
from src.utils.io import load_results
from src.analysis.question_execution_time import hard_tier_question_pool, MODEL_LABELS
from src.analysis.error_definitions import ERROR_TYPE_DEFINITIONS


def balanced_question_pool(df: pd.DataFrame) -> dict:
    """Same 40/40/40 pool as question_execution_time.py / error_heatmap.py."""

    return {
        "Easy": sorted(df[df["Difficulty"] == "Easy"]["Question ID"].unique()),
        "Medium": sorted(df[df["Difficulty"] == "Medium"]["Question ID"].unique()),
        "Hard": hard_tier_question_pool(df),
    }


def error_type_difficulty_table(df: pd.DataFrame, model: str, prompt: str, pool: dict, error_type_order: list) -> pd.DataFrame:
    """Error Type x Difficulty counts + Total, for one model/prompt, balanced pool only."""

    rows = df[
        (df["Model"] == model) & (df["Prompt"] == prompt)
        & df.apply(lambda r: r["Question ID"] in pool[r["Difficulty"]], axis=1)
    ]
    wrong = rows[rows["Answer Correct"] == False]  # noqa: E712

    data = []
    for t in error_type_order:
        counts = [int(((wrong["Difficulty"] == d) & (wrong["Error Type"] == t)).sum()) for d in DIFFICULTIES]
        total = sum(counts)
        if total == 0:
            continue
        data.append([t] + counts + [total])

    total_row = ["Total incorrect"] + [int((wrong["Difficulty"] == d).sum()) for d in DIFFICULTIES] + [len(wrong)]
    data.append(total_row)

    return pd.DataFrame(data, columns=["Error Type"] + DIFFICULTIES + ["Total"])


def render_table_image(ax, table_df: pd.DataFrame, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(table_df.columns))))
    tbl.scale(1, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dde4f0")
            cell.set_text_props(fontweight="bold")
        elif row == len(table_df):
            cell.set_facecolor("#eef0f3")
            cell.set_text_props(fontweight="bold")
        if col == 0:
            cell.set_text_props(ha="left")
    ax.axis("tight")


def render_glossary_panel(ax) -> None:
    """Error-type meanings, embedded in the image instead of a separate file."""

    ax.axis("off")
    ax.set_title("What each error type means", fontsize=11, fontweight="bold", loc="left")
    lines = [f"{t}  -  {meaning}" for t, meaning in ERROR_TYPE_DEFINITIONS.items()]
    mid = (len(lines) + 1) // 2
    left_col = "\n".join(lines[:mid])
    right_col = "\n".join(lines[mid:])
    ax.text(0.0, 0.85, left_col, transform=ax.transAxes, fontsize=9, ha="left", va="top")
    ax.text(0.52, 0.85, right_col, transform=ax.transAxes, fontsize=9, ha="left", va="top")


def plot_prompt_tables(prompt: str, df: pd.DataFrame, pool: dict, error_type_order: list) -> None:
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, len(MODEL_NAMES), height_ratios=[3, 1.6], hspace=0.35)

    for col, model in enumerate(MODEL_NAMES):
        table_df = error_type_difficulty_table(df, model, prompt, pool, error_type_order)

        os.makedirs(TABLES_DIR, exist_ok=True)
        csv_path = os.path.join(TABLES_DIR, f"error_type_distribution_{model}_{prompt}.csv")
        table_df.to_csv(csv_path, index=False)
        print(f"Saved -> {csv_path}")

        ax = fig.add_subplot(gs[0, col])
        render_table_image(ax, table_df, MODEL_LABELS[model])

    glossary_ax = fig.add_subplot(gs[1, :])
    render_glossary_panel(glossary_ax)

    fig.suptitle(
        f"RQ1 - Distribution of Reasoning Error Types ({prompt.upper()}) - "
        f"40 Easy + 40 Medium + 40 Hard = 120 questions",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"error_type_distribution_by_model_{prompt}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def run():
    df = load_results()
    df = df.copy()
    df["Answer Correct"] = df["Answer Correct"].astype(str).str.lower() == "true"

    pool = balanced_question_pool(df)

    selected = df[df.apply(lambda r: r["Question ID"] in pool[r["Difficulty"]], axis=1)]
    wrong_selected = selected[selected["Answer Correct"] == False]  # noqa: E712
    error_type_order = wrong_selected["Error Type"].value_counts().index.tolist()

    for prompt in PROMPT_TYPES:
        plot_prompt_tables(prompt, df, pool, error_type_order)


if __name__ == "__main__":
    run()
