import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# FILE PATHS
# ==========================================================

RESULTS = "data/results/experiment_results.csv"

OUTPUT_FOLDER = "outputs/figures"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LOAD RESULTS
# ==========================================================

df = pd.read_csv(RESULTS)

# ==========================================================
# CALCULATE ACCURACY
# ==========================================================

accuracy = (

    df.groupby(["Model", "Difficulty"])["Answer Correct"]

      .mean()

      .mul(100)

      .unstack()

)

# Ensure order
accuracy = accuracy.loc[
    ["gemini", "groq", "mistral"],
    ["Hard"]
]

# ==========================================================
# COLOURS
# ==========================================================

model_colors = {

    "gemini": "#4F81BD",      # Blue

    "groq": "#70AD47",        # Green

    "mistral": "#ED7D31"      # Orange

}

# ==========================================================
# PLOT
# ==========================================================

fig, ax = plt.subplots(figsize=(12,6))

group_width = 4

bar_width = 0.8

models = accuracy.index.tolist()

for i, model in enumerate(models):

    values = accuracy.loc[model].values

    #positions = np.arange(3) + i * group_width

    bars = ax.bar(

        #positions,

        values,

        width=bar_width,

        color=model_colors[model],

        edgecolor="black"

    )

    # Accuracy labels
    for bar in bars:

        height = bar.get_height()

        ax.text(

            bar.get_x() + bar.get_width()/2,

            height + 1,

            f"{height:.1f}%",

            ha="center",

            fontsize=10,

            weight="bold"

        )

# ==========================================================
# X LABELS
# ==========================================================

xticks = []

xticklabels = []

for i in range(3):

    start = i * group_width

    xticks.extend([

        start,

        start + 1,

        start + 2

    ])

    xticklabels.extend([

        "Easy",

        "Medium",

        "Hard"

    ])

ax.set_xticks(xticks)

ax.set_xticklabels(

    xticklabels,

    fontsize=11

)

# ==========================================================
# MODEL NAMES
# ==========================================================

for i, model in enumerate(models):

    center = i * group_width + 1

    ax.text(

        center,

        -12,

        model.capitalize(),

        ha="center",

        fontsize=13,

        weight="bold"

    )

# ==========================================================
# AXES
# ==========================================================

ax.set_ylabel(

    "Accuracy (%)",

    fontsize=13,

    weight="bold"

)

ax.set_ylim(0,110)

ax.set_title(

    "Model Accuracy Across Difficulty Levels",

    fontsize=18,

    weight="bold"

)

ax.grid(

    axis="y",

    linestyle="--",

    alpha=0.4

)

# ==========================================================
# SAVE
# ==========================================================

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "model_accuracy.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.show()

print("="*60)
print("Model Accuracy Figure Created")
print("="*60)