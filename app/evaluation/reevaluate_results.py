"""
Re-evaluate experiment results using the updated evaluator.

This script DOES NOT call any LLM.

It simply:
1. Loads experiment_results.csv
2. Recomputes Answer Correct
3. Saves CSV
4. Regenerates Excel
"""

import pandas as pd
import os
from app.config import RESULTS_PATH
from app.evaluation.evaluator import evaluate_response
from app.utils.csv_to_excel import csv_to_excel


def reevaluate():

    print("=" * 60)
    print("Loading Results")
    print("=" * 60)

    results = pd.read_csv(RESULTS_PATH)

    corrected = 0

    for index, row in results.iterrows():

        evaluation = evaluate_response(

            row["Ground Truth Answer"],

            row["Model Final Answer"]

        )

        old_value = bool(row["Answer Correct"])
        new_value = evaluation["correct"]

        if old_value != new_value:
            corrected += 1

        results.at[index, "Answer Correct"] = new_value

    print(f"\nSaving CSV to:\n{os.path.abspath(RESULTS_PATH)}")

    results.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("CSV saved successfully.")

    csv_to_excel(RESULTS_PATH)

    print()
    print("=" * 60)
    print("Re-evaluation Completed")
    print("=" * 60)
    print(f"Rows Updated : {corrected}")
    print(f"Total Rows   : {len(results)}")
    print("=" * 60)


if __name__ == "__main__":

    reevaluate()