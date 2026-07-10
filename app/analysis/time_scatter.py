import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors

# ==========================================================
# Configuration
# ==========================================================

RESULT_FILE = "data/results/experiment_results.csv"
OUTPUT_FOLDER = "outputs/figures"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# Load Results
# ==========================================================

df = pd.read_csv(RESULT_FILE)

TIME_COLUMN = "Execution Time (s)"

difficulty_order = ["Easy", "Medium", "Hard"]

df["Difficulty"] = pd.Categorical(
    df["Difficulty"],
    categories=difficulty_order,
    ordered=True
)

# ==========================================================
# Plot Settings
# ==========================================================

base_positions = {
    "Easy": 1,
    "Medium": 2,
    "Hard": 3
}

offsets = {
    "gemini": -0.18,
    "groq": 0,
    "mistral": 0.18
}

colors = {
    "gemini": "#1f77b4",
    "groq": "#2ca02c",
    "mistral": "#ff9900"
}

plt.figure(figsize=(10,6))

scatter_info = []

# ==========================================================
# Scatter Plot
# ==========================================================

for model in ["gemini", "groq", "mistral"]:

    model_df = df[df["Model"].str.lower() == model]

    x = [
        base_positions[d]
        + offsets[model]
        + np.random.uniform(-0.05, 0.05)
        for d in model_df["Difficulty"]
    ]

    scatter = plt.scatter(
        x,
        model_df[TIME_COLUMN],
        s=90,
        color=colors[model],
        label=model.capitalize()
    )

    scatter_info.append((scatter, model_df.reset_index(drop=True)))

# ==========================================================
# Hover Information
# ==========================================================

for scatter, info in scatter_info:

    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(sel, info=info):

        row = info.iloc[sel.index]

        sel.annotation.set_text(
            f"Question : Q{row['Question Number']}\n"
            f"Model : {row['Model']}\n"
            f"Prompt : {row['Prompt']}\n"
            f"Difficulty : {row['Difficulty']}\n"
            f"Time : {row[TIME_COLUMN]:.2f} sec\n"
            f"Correct : {row['Answer Correct']}"
        )

# ==========================================================
# Formatting
# ==========================================================

plt.xticks(
    [1,2,3],
    ["Easy","Medium","Hard"],
    fontsize=12
)

plt.xlabel(
    "Question Difficulty",
    fontsize=13
)

plt.ylabel(
    "Execution Time (seconds)",
    fontsize=13
)

plt.title(
    "Execution Time by Model and Question Difficulty",
    fontsize=15,
    weight="bold"
)

plt.grid(axis="y", alpha=0.3)

plt.legend(fontsize=11)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/time_scatter.png",
    dpi=300
)

plt.show()