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
# HARD QUESTIONS ONLY
# ==========================================================

hard = results[
    results["Difficulty"] == "Hard"
].copy()

print()
print("=" * 60)
print("Hard Question Summary")
print("=" * 60)

print("Total Hard Experiments :", len(hard))

# ==========================================================
# MODEL ACCURACY
# ==========================================================

accuracy = (

    hard

    .groupby("Model")["Answer Correct"]

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

print()

print("=" * 60)
print("Hard Question Accuracy")
print("=" * 60)

print(accuracy)

# ==========================================================
# SAVE TABLE
# ==========================================================

accuracy.to_csv(

    os.path.join(

        TABLE_FOLDER,

        "hard_model_accuracy.csv"

    ),

    index=False

)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(7,6))

bars = plt.bar(

    accuracy["Model"],

    accuracy["Accuracy (%)"],

    width=0.6

)

plt.title(

    "Model Accuracy on Hard Questions"

)

plt.xlabel(

    "Model"

)

plt.ylabel(

    "Accuracy (%)"

)

plt.ylim(0,100)

# ==========================================================
# VALUE LABELS
# ==========================================================

for bar in bars:

    plt.text(

        bar.get_x() + bar.get_width()/2,

        bar.get_height() + 1,

        f"{bar.get_height():.2f}%",

        ha="center",

        fontsize=10

    )

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURE_FOLDER,

        "hard_model_accuracy.png"

    ),

    dpi=300

)

plt.close()

# ==========================================================
# BEST MODEL
# ==========================================================

best = accuracy.sort_values(

    "Accuracy (%)",

    ascending=False

).iloc[0]

print()

print("=" * 60)
print("Best Performing Model")
print("=" * 60)

print(best)

print()

print("=" * 60)
print("Figure Saved")
print("=" * 60)

print(

    os.path.join(

        FIGURE_FOLDER,

        "hard_model_accuracy.png"

    )

)