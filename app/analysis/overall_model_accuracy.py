
import os
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================================
# FILE PATHS
# ==========================================================

RESULTS_FILE = "data/results/experiment_results.csv"

OUTPUT_FOLDER = "outputs"

FIGURE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "figures"
)

TABLE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "tables"
)

os.makedirs(FIGURE_FOLDER, exist_ok=True)
os.makedirs(TABLE_FOLDER, exist_ok=True)

# ==========================================================
# LOAD RESULTS
# ==========================================================

print("=" * 60)
print("Loading Experiment Results")
print("=" * 60)

results = pd.read_csv(RESULTS_FILE)

# ==========================================================
# CALCULATE ACCURACY
# ==========================================================

accuracy = (
    results
    .groupby(["Difficulty", "Model"])["Answer Correct"]
    .mean()
    .mul(100)
    .round(2)
    .reset_index()
)

accuracy.rename(
    columns={
        "Answer Correct": "Accuracy (%)"
    },
    inplace=True
)

# ==========================================================
# SAVE TABLE
# ==========================================================

accuracy.to_csv(
    os.path.join(
        TABLE_FOLDER,
        "model_accuracy_by_difficulty.csv"
    ),
    index=False
)

# ==========================================================
# PREPARE DATA
# ==========================================================

pivot = accuracy.pivot(
    index="Difficulty",
    columns="Model",
    values="Accuracy (%)"
)

pivot = pivot.reindex(
    ["Easy", "Medium", "Hard"]
)

# ==========================================================
# PLOT
# ==========================================================

ax = pivot.plot(
    kind="bar",
    figsize=(9,6),
    width=0.75
)

plt.title(
    "Model Accuracy Across Difficulty Levels",
    fontsize=15
)

plt.xlabel(
    "Difficulty",
    fontsize=12
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=12
)

plt.ylim(0,100)

plt.xticks(rotation=0)

plt.legend(
    title="Model"
)

# ==========================================================
# VALUE LABELS
# ==========================================================

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_FOLDER,
        "model_accuracy_by_difficulty.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SUMMARY
# ==========================================================
# ==========================================================
# OVERALL MODEL ACCURACY
# ==========================================================

overall = (

    results

    .groupby("Model")["Answer Correct"]

    .mean()

    .mul(100)

    .round(2)

    .reset_index()

)

overall.rename(

    columns={

        "Answer Correct": "Overall Accuracy (%)"

    },

    inplace=True

)

overall.to_csv(

    os.path.join(

        TABLE_FOLDER,

        "overall_model_accuracy.csv"

    ),

    index=False

)

print()

print("=" * 60)
print("Overall Model Accuracy")
print("=" * 60)
print(overall)


print()
print("=" * 60)
print("Accuracy Table")
print("=" * 60)
print(accuracy)

print()
print("=" * 60)
print("Figure Saved")
print("=" * 60)

print(
    os.path.join(
        FIGURE_FOLDER,
        "model_accuracy_by_difficulty.png"
    )
)