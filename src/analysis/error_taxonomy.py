"""
RQ1: error-type distribution + chi-square/Fisher's exact independence
test on Model x Error Type.
"""

import os

import pandas as pd

from config.config import TABLES_DIR, MODEL_NAMES
from src.utils.io import load_results
from src.utils.stats import independence_test
from src.analysis.question_execution_time import hard_tier_question_pool


def error_type_by_model_contingency() -> pd.DataFrame:

    df = load_results()
    pool = hard_tier_question_pool(df)
    df = df[(df["Difficulty"] != "Hard") | (df["Question ID"].isin(pool))]
    errors = df[df["Error Type"].astype(str).str.strip() != "Correct"]

    table = pd.crosstab(errors["Model"], errors["Error Type"])
    table = table.reindex(index=[m for m in MODEL_NAMES if m in table.index])

    return table


def run():

    os.makedirs(TABLES_DIR, exist_ok=True)

    contingency = error_type_by_model_contingency()
    contingency_path = os.path.join(TABLES_DIR, "rq1_error_type_by_model.csv")
    contingency.to_csv(contingency_path)
    print(f"Saved -> {contingency_path}")
    print(contingency)
    print()

    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        test = independence_test(contingency)
        test_df = pd.DataFrame([test])
        test_path = os.path.join(TABLES_DIR, "rq1_independence_test.csv")
        test_df.to_csv(test_path, index=False)
        print(f"Saved -> {test_path}")
        print(test)
    else:
        print("Not enough model/error-type variety yet for an independence test.")

    return contingency


if __name__ == "__main__":
    run()
