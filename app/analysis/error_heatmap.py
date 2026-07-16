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
# KEEP ONLY INCORRECT ANSWERS
# ==========================================================

errors = results[
    results["Answer Correct"] == False
].copy()

# ==========================================================
# CREATE HEATMAP TABLE
# ==========================================================

heatmap = pd.crosstab(

    errors["Model"],

    errors["Error Type"]

)

# Save table

heatmap.to_csv(

    os.path.join(

        TABLE_FOLDER,

        "error_heatmap.csv"

    )

)

print()

print("=" * 60)
print("Error Heatmap Table")
print("=" * 60)

print(heatmap)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(12,6))

plt.imshow(

    heatmap,
    aspect="auto",
    cmap="Reds",
    interpolation="nearest"

)

# ==========================================================
# AXES
# ==========================================================

plt.xticks(

    range(len(heatmap.columns)),

    heatmap.columns,

    rotation=45,

    ha="right"

)

plt.yticks(

    range(len(heatmap.index)),

    heatmap.index

)

plt.xlabel(

    "Error Type"

)

plt.ylabel(

    "Model"

)

plt.title(

    "Reasoning Error Heatmap by Model"

)

# ==========================================================
# CELL VALUES
# ==========================================================

max_value = heatmap.values.max()

for i in range(len(heatmap.index)):

    for j in range(len(heatmap.columns)):

        value = heatmap.iloc[i, j]

        text_color = "white" if value > max_value / 2 else "black"

        plt.text(

            j,

            i,

            str(value),

            ha="center",

            va="center",

            fontsize=9,

            color=text_color,

            fontweight="bold"

        )

# ==========================================================
# COLOR BAR
# ==========================================================

plt.colorbar(
    label="Number of Errors"
)

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURE_FOLDER,

        "error_heatmap.png"

    ),

    dpi=300

)

plt.close()

# ==========================================================
# SUMMARY
# ==========================================================

print()

print("=" * 60)
print("Figure Saved")
print("=" * 60)

print(

    os.path.join(

        FIGURE_FOLDER,

        "error_heatmap.png"

    )

)