"""
RQ2: At what step in the reasoning chain do errors occur?

This module didn't exist at all in the prior pipeline — the judge already
collected `Error Step`, but nothing ever analyzed it, leaving one of only
three research questions with zero supporting analysis despite having the
underlying data. This is the fix.

Raw step number isn't comparable across responses of different lengths
("error at step 3 of 4" and "error at step 3 of 12" mean very different
things), so the primary measure here is the NORMALIZED error position:
Error Step / Step Count, in [0, 1] — how far through the response the
first error occurred, independent of response length.
"""

import os

import pandas as pd

from config.config import TABLES_DIR, MODEL_NAMES, PROMPT_TYPES
from src.utils.io import load_results
from src.utils.stats import kruskal_wallis


def normalized_error_positions() -> pd.DataFrame:

    df = load_results()

    errors = df[df["Error Type"].astype(str).str.strip() != "Correct"].copy()

    errors["Error Step"] = pd.to_numeric(errors["Error Step"], errors="coerce")
    errors["Step Count"] = pd.to_numeric(errors["Step Count"], errors="coerce")

    errors = errors[
        errors["Error Step"].notna()
        & errors["Step Count"].notna()
        & (errors["Step Count"] > 0)
    ].copy()

    errors["Normalized Error Position"] = (
        errors["Error Step"] / errors["Step Count"]
    ).clip(0, 1)

    return errors


def summary_by_model_prompt(errors: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for model in MODEL_NAMES:
        for prompt in PROMPT_TYPES:

            subset = errors[(errors["Model"] == model) & (errors["Prompt"] == prompt)]

            if len(subset) == 0:
                continue

            rows.append({
                "Model": model,
                "Prompt": prompt,
                "N Errors": len(subset),
                "Mean Normalized Position": subset["Normalized Error Position"].mean(),
                "Median Normalized Position": subset["Normalized Error Position"].median(),
                "Mean Raw Error Step": subset["Error Step"].mean(),
            })

    return pd.DataFrame(rows)


def run():

    os.makedirs(TABLES_DIR, exist_ok=True)

    errors = normalized_error_positions()

    detail_path = os.path.join(TABLES_DIR, "rq2_error_position_detail.csv")
    errors.to_csv(detail_path, index=False)
    print(f"Saved -> {detail_path}")

    summary = summary_by_model_prompt(errors)
    summary_path = os.path.join(TABLES_DIR, "rq2_error_position_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved -> {summary_path}")
    print(summary)
    print()

    # Does error position differ significantly across models?
    groups = [
        errors[errors["Model"] == model]["Normalized Error Position"].dropna().tolist()
        for model in MODEL_NAMES
    ]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) >= 2:
        test = kruskal_wallis(*groups)
        test_df = pd.DataFrame([{"Comparison": "Error position across models", **test}])
        test_path = os.path.join(TABLES_DIR, "rq2_kruskal_wallis.csv")
        test_df.to_csv(test_path, index=False)
        print(f"Saved -> {test_path}")
        print(test)
    else:
        print("Not enough model variety yet for a Kruskal-Wallis test.")

    return errors, summary


if __name__ == "__main__":
    run()
