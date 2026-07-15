import pandas as pd

# ==========================================================
# FILE PATHS
# ==========================================================

FINAL_DATASET = "data/processed/final_dataset.xlsx"

RESULTS = "data/results/experiment_results.csv"

# ==========================================================
# LOAD FILES
# ==========================================================

dataset = pd.read_excel(FINAL_DATASET)

results = pd.read_csv(RESULTS)

# ==========================================================
# LOOKUPS
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
# UPDATE ONLY MISSING VALUES
# ==========================================================

mask = (
    results["Ground Truth Final Answer"].isna()
    | (results["Ground Truth Final Answer"].astype(str).str.strip() == "")
)

results.loc[mask, "Ground Truth Final Answer"] = (
    results.loc[mask, "Question ID"].map(answer_lookup)
)

mask2 = (
    results["Ground Truth Solution"].isna()
    | (results["Ground Truth Solution"].astype(str).str.strip() == "")
)

results.loc[mask2, "Ground Truth Solution"] = (
    results.loc[mask2, "Question ID"].map(solution_lookup)
)

# ==========================================================
# SAVE
# ==========================================================

results.to_csv(
    RESULTS,
    index=False
)

print("✅ Experiment results updated successfully.")