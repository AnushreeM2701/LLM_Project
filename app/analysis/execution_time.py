import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# Configuration
# ==========================================================

RESULT_FILE = "data/results/experiment_results.csv"
OUTPUT_FOLDER = "outputs/figures"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# Load data
# ==========================================================

df = pd.read_csv(RESULT_FILE)

TIME_COLUMN = "Execution Time (s)"

# ==========================================================
# Average execution time
# ==========================================================

avg_time = (
    df.groupby(["Prompt", "Model"])[TIME_COLUMN]
      .mean()
      .reset_index()
)

prompt_order = ["baseline", "cot", "tot"]
model_order = ["gemini", "groq", "mistral"]

pivot = (
    avg_time
    .pivot(index="Prompt", columns="Model", values=TIME_COLUMN)
    .reindex(prompt_order)
)

pivot = pivot[model_order]

# ==========================================================
# Plot
# ==========================================================

x = np.arange(len(prompt_order))
width = 0.25

fig, ax = plt.subplots(figsize=(9,6))

bars1 = ax.bar(
    x-width,
    pivot["gemini"],
    width,
    label="Gemini"
)

bars2 = ax.bar(
    x,
    pivot["groq"],
    width,
    label="Groq"
)

bars3 = ax.bar(
    x+width,
    pivot["mistral"],
    width,
    label="Mistral"
)

# ----------------------------------------------------------

ax.set_xticks(x)
ax.set_xticklabels(["Baseline","CoT","ToT"], fontsize=11)

ax.set_ylabel("Average Execution Time (seconds)", fontsize=12)

ax.set_xlabel("Prompting Technique", fontsize=12)

ax.set_title(
    "Average Execution Time by Model and Prompt",
    fontsize=14,
    weight="bold"
)

ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend()

# ==========================================================
# Add values on bars
# ==========================================================

def add_labels(bars):

    for bar in bars:

        height = bar.get_height()

        ax.text(

            bar.get_x()+bar.get_width()/2,

            height+0.3,

            f"{height:.1f}",

            ha="center",

            fontsize=10

        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/execution_time_grouped.png",
    dpi=300
)

plt.show()