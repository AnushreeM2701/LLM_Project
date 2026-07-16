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

os.makedirs(FIGURE_FOLDER, exist_ok=True)

# ==========================================================
# LOAD RESULTS
# ==========================================================

print("=" * 60)
print("Loading Experiment Results")
print("=" * 60)

results = pd.read_csv(RESULTS_FILE)

# ==========================================================
# OVERALL PROMPT ACCURACY
# ==========================================================

overall = (

    results

    .groupby("Prompt")["Answer Correct"]

    .mean()

    .mul(100)

    .round(2)

    .reset_index()

)

overall.rename(

    columns={

        "Answer Correct": "Accuracy (%)"

    },

    inplace=True

)

# Make sure CoT appears before ToT

prompt_order = ["cot", "tot"]

overall["Prompt"] = pd.Categorical(

    overall["Prompt"],

    categories=prompt_order,

    ordered=True

)

overall = overall.sort_values("Prompt")

print()

print("=" * 60)
print("Overall Prompt Accuracy")
print("=" * 60)

print(overall)

# ==========================================================
# BAR CHART
# ==========================================================

plt.figure(figsize=(6,6))

bars = plt.bar(

    ["CoT", "ToT"],

    overall["Accuracy (%)"],

    width=0.6

)

plt.title(

    "Overall Prompt Accuracy"

)

plt.xlabel(

    "Prompt"

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

        "overall_prompt_accuracy.png"

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

        "overall_prompt_accuracy.png"

    )

)