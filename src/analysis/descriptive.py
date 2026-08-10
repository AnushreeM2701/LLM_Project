"""
Descriptive accuracy tables with Wilson confidence intervals.
"""

import os

import pandas as pd

from config.config import TABLES_DIR, MODEL_NAMES, PROMPT_TYPES
from src.utils.io import load_results
from src.utils.stats import wilson_ci
from src.analysis.question_execution_time import hard_tier_question_pool


def _restrict_hard_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Easy/Medium as-is; restrict Hard to the seeded 40-question balanced pool."""

    pool = hard_tier_question_pool(df)
    is_hard = df["Difficulty"] == "Hard"
    return df[(~is_hard) | (is_hard & df["Question ID"].isin(pool))]


def _accuracy_row(df: pd.DataFrame, group_desc: dict) -> dict:

    n = len(df)
    correct = df["Answer Correct"].astype(bool).sum()
    ci = wilson_ci(int(correct), n)

    return {
        **group_desc,
        "N": n,
        "Correct": int(correct),
        "Accuracy": ci["proportion"],
        "CI Lower": ci["lower"],
        "CI Upper": ci["upper"],
    }


def accuracy_by_model_prompt_difficulty() -> pd.DataFrame:

    df = _restrict_hard_tier(load_results())
    rows = []

    for model in MODEL_NAMES:
        for prompt in PROMPT_TYPES:
            for difficulty in ["Easy", "Medium", "Hard"]:
                subset = df[
                    (df["Model"] == model)
                    & (df["Prompt"] == prompt)
                    & (df["Difficulty"] == difficulty)
                ]
                if len(subset) == 0:
                    continue
                rows.append(_accuracy_row(
                    subset, {"Model": model, "Prompt": prompt, "Difficulty": difficulty}
                ))

    return pd.DataFrame(rows)


def accuracy_by_model_category() -> pd.DataFrame:

    df = _restrict_hard_tier(load_results())
    rows = []

    for model in MODEL_NAMES:
        for category in sorted(df["Category"].dropna().unique()):
            for difficulty in ["Easy", "Medium", "Hard"]:
                for prompt in PROMPT_TYPES:
                    subset = df[
                        (df["Model"] == model)
                        & (df["Category"] == category)
                        & (df["Difficulty"] == difficulty)
                        & (df["Prompt"] == prompt)
                    ]
                    if len(subset) == 0:
                        continue
                    rows.append(_accuracy_row(
                        subset,
                        {"Model": model, "Category": category, "Difficulty": difficulty, "Prompt": prompt},
                    ))

    return pd.DataFrame(rows)


def run():

    os.makedirs(TABLES_DIR, exist_ok=True)

    tables = {
        "accuracy_by_model_prompt_difficulty.csv": accuracy_by_model_prompt_difficulty(),
        "accuracy_by_model_category.csv": accuracy_by_model_category(),
    }

    for filename, table in tables.items():
        path = os.path.join(TABLES_DIR, filename)
        table.to_csv(path, index=False)
        print(f"Saved -> {path}")
        print(table)
        print()

    return tables


if __name__ == "__main__":
    run()
