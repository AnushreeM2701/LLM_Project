import os
import matplotlib.pyplot as plt
import pandas as pd

# FILE PATHS

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

# LOAD RESULTS

print("=" * 60)
print("Loading Experiment Results")
print("=" * 60)

results = pd.read_csv(RESULTS_FILE)

# KEEP ONLY INCORRECT ANSWERS

errors = results[
    results["Answer Correct"] == False
].copy()

# ERROR COUNTS

error_counts = (
    errors["Error Type"]
    .value_counts()
    .reset_index()
)

error_counts.columns = [
    "Error Type",
    "Count"
]

print()

print("=" * 60)
print("Error Distribution")
print("=" * 60)

print(error_counts)

# SAVE TABLE

error_counts.to_csv(

    os.path.join(

        TABLE_FOLDER,

        "error_distribution.csv"

    ),

    index=False

)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(

    figsize=(10,6)

)

bars = plt.barh(

    error_counts["Error Type"],

    error_counts["Count"]

)

plt.title(

    "Distribution of Reasoning Error Types"

)

plt.xlabel(

    "Number of Errors"

)

plt.ylabel(

    "Error Type"

)

plt.gca().invert_yaxis()

# ==========================================================
# VALUE LABELS
# ==========================================================

for bar in bars:

    width = bar.get_width()

    plt.text(

        width + 1,

        bar.get_y() + bar.get_height()/2,

        str(int(width)),

        va="center",

        fontsize=10

    )

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURE_FOLDER,

        "error_distribution.png"

    ),

    dpi=300

)

plt.close()

# ==========================================================
# SUMMARY
# ==========================================================

print()

print("=" * 60)
print("Most Common Error")
print("=" * 60)

print(error_counts.iloc[0])

print()

print("=" * 60)
print("Figure Saved")
print("=" * 60)

print(

    os.path.join(

        FIGURE_FOLDER,

        "error_distribution.png"

    )

)