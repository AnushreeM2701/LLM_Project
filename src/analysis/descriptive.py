"""
Descriptive accuracy tables with Wilson confidence intervals.
"""

import os

import pandas as pd

from config.config import TABLES_DIR, MODEL_NAMES, PROMPT_TYPES
from src.utils.io import load_results
from src.utils.stats import wilson_ci


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


def accuracy_by_model_prompt() -> pd.DataFrame:

    df = load_results()
    rows = []

    for model in MODEL_NAMES:
        for prompt in PROMPT_TYPES:
            subset = df[(df["Model"] == model) & (df["Prompt"] == prompt)]
            if len(subset) == 0:
                continue
            rows.append(_accuracy_row(subset, {"Model": model, "Prompt": prompt}))

    return pd.DataFrame(rows)


def accuracy_by_model_prompt_difficulty() -> pd.DataFrame:

    df = load_results()
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

    df = load_results()
    rows = []

    for model in MODEL_NAMES:
        for category in sorted(df["Category"].dropna().unique()):
            subset = df[(df["Model"] == model) & (df["Category"] == category)]
            if len(subset) == 0:
                continue
            rows.append(_accuracy_row(subset, {"Model": model, "Category": category}))

    return pd.DataFrame(rows)


def run():

    os.makedirs(TABLES_DIR, exist_ok=True)

    tables = {
        "accuracy_by_model_prompt.csv": accuracy_by_model_prompt(),
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
