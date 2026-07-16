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
# KEEP ONLY HARD QUESTIONS
# ==========================================================

hard = results[
    results["Difficulty"] == "Hard"
].copy()

# ==========================================================
# CALCULATE ACCURACY
# ==========================================================

accuracy = (

    hard

    .groupby(
        ["Model", "Prompt"]
    )["Answer Correct"]

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
print("Hard Question Prompt Accuracy")
print("=" * 60)

print(accuracy)

# ==========================================================
# SAVE TABLE
# ==========================================================

accuracy.to_csv(

    os.path.join(

        TABLE_FOLDER,

        "hard_prompt_accuracy.csv"

    ),

    index=False

)

# ==========================================================
# PREPARE DATA
# ==========================================================

pivot = accuracy.pivot(

    index="Model",

    columns="Prompt",

    values="Accuracy (%)"

)

pivot = pivot[
    [
        "cot",
        "tot"
    ]
]

# ==========================================================
# PLOT
# ==========================================================

ax = pivot.plot(

    kind="bar",

    figsize=(8,6),

    width=0.70

)

plt.title(

    "Prompt Accuracy on Hard Questions"

)

plt.xlabel(

    "Model"

)

plt.ylabel(

    "Accuracy (%)"

)

plt.ylim(0,100)

plt.xticks(rotation=0)

plt.legend(

    title="Prompt",

    labels=["CoT", "ToT"]

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

        "hard_prompt_accuracy.png"

    ),

    dpi=300

)

plt.close()

print()

print("=" * 60)
print("Figure Saved")
print("=" * 60)

print(

    os.path.join(

        FIGURE_FOLDER,

        "hard_prompt_accuracy.png"

    )

)