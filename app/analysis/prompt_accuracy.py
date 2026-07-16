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
# CALCULATE PROMPT ACCURACY
# ==========================================================

prompt_accuracy = (

    results

    .groupby(
        ["Model", "Prompt"]
    )["Answer Correct"]

    .mean()

    .mul(100)

    .round(2)

    .reset_index()

)

prompt_accuracy.rename(

    columns={

        "Answer Correct": "Accuracy (%)"

    },

    inplace=True

)

print()

print("=" * 60)
print("Prompt Accuracy")
print("=" * 60)

print(prompt_accuracy)

# ==========================================================
# SAVE TABLE
# ==========================================================

prompt_accuracy.to_csv(

    os.path.join(

        TABLE_FOLDER,

        "prompt_accuracy.csv"

    ),

    index=False

)

# ==========================================================
# PREPARE DATA
# ==========================================================

pivot = prompt_accuracy.pivot(

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
# BAR CHART
# ==========================================================

ax = pivot.plot(

    kind="bar",

    figsize=(8,6),

    width=0.70

)

plt.title(

    "Prompt Accuracy by Model"

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

        "prompt_accuracy.png"

    ),

    dpi=300

)

plt.close()

# ==========================================================
# SUMMARY
# ==========================================================

print()

print("=" * 60)
print("Average Prompt Accuracy")
print("=" * 60)

print(

    results

    .groupby("Prompt")["Answer Correct"]

    .mean()

    .mul(100)

    .round(2)

)

print()

print("=" * 60)
print("Figure Saved")
print("=" * 60)

print(

    os.path.join(

        FIGURE_FOLDER,

        "prompt_accuracy.png"

    )

)