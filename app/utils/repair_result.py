import pandas as pd
from openpyxl.styles import Alignment

from app.parser.response_parser import parse_response
from app.evaluation.evaluator import evaluate_response

# ==========================================================
# FILE PATHS
# ==========================================================

FINAL_DATASET = "data/processed/final_dataset.xlsx"

RESULTS_CSV = "data/results/experiment_results.csv"

RESULTS_EXCEL = "data/results/experiment_results.xlsx"

# ==========================================================
# LOAD FILES
# ==========================================================

print("Loading Final Dataset...")

dataset = pd.read_excel(FINAL_DATASET)

print("Loading Experiment Results...")

results = pd.read_csv(RESULTS_CSV)
text_columns = [
    "Ground Truth Final Answer",
    "Ground Truth Solution",
    "Model Final Answer",
    "Model Response",
    "Error Type"
]

for col in text_columns:

    if col in results.columns:

        results[col] = (
            results[col]
            .fillna("")
            .astype(str)
        )

# ==========================================================
# CREATE LOOKUPS
# ==========================================================

answer_lookup = dict(
    zip(
        dataset["Question ID"],
        dataset["Ground Truth Final Answer"]
    )
)

solution_lookup = dict(
    zip(
        dataset["Question ID"],
        dataset["Ground Truth Solution"]
    )
)

# ==========================================================
# REPAIR RESULTS
# ==========================================================

print("Repairing experiment results...")

for index, row in results.iterrows():

    question_id = row["Question ID"]

    model_response = str(row["Model Response"])

    # ------------------------------------------
    # Re-parse model response
    # ------------------------------------------

    parsed = parse_response(model_response)

    results.at[index, "Model Final Answer"] = (
        parsed["model_final_answer"]
    )

    # ------------------------------------------
    # Update Ground Truth
    # ------------------------------------------

    ground_truth_answer = answer_lookup.get(
        question_id,
        ""
    )

    ground_truth_solution = solution_lookup.get(
        question_id,
        ""
    )

    results.at[index, "Ground Truth Final Answer"] = (
        ground_truth_answer
    )

    results.at[index, "Ground Truth Solution"] = (
        ground_truth_solution
    )

    # ------------------------------------------
    # Re-evaluate
    # ------------------------------------------

    evaluation = evaluate_response(
        ground_truth_answer,
        parsed["model_final_answer"]
    )

    results.at[index, "Answer Correct"] = (
        evaluation["answer_correct"]
    )

    results.at[index, "Error Type"] = (
        evaluation["error_type"]
    )

print("Repair completed.")

# ==========================================================
# SAVE CSV
# ==========================================================

results.to_csv(
    RESULTS_CSV,
    index=False
)

# ==========================================================
# SAVE EXCEL
# ==========================================================

with pd.ExcelWriter(
    RESULTS_EXCEL,
    engine="openpyxl"
) as writer:

    results.to_excel(
        writer,
        index=False,
        sheet_name="Experiment Results"
    )

    worksheet = writer.sheets["Experiment Results"]

    worksheet.freeze_panes = "A2"

    worksheet.auto_filter.ref = worksheet.dimensions

    for row in worksheet.iter_rows():

        max_lines = 1

        for cell in row:

            cell.alignment = Alignment(
                wrap_text=True,
                vertical="top"
            )

            if cell.value:

                max_lines = max(
                    max_lines,
                    str(cell.value).count("\n") + 1
                )

        worksheet.row_dimensions[row[0].row].height = max(
            25,
            max_lines * 18
        )

print()
print("=" * 60)
print("Experiment Results Repaired Successfully")
print("=" * 60)
print("CSV   :", RESULTS_CSV)
print("Excel :", RESULTS_EXCEL)