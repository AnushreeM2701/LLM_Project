import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
# FIND EXECUTION TIME COLUMN
# ==========================================================

possible_columns = [

    "Execution Time (s)",

    "Execution Time",

    "Time Taken",

    "Response Time",

    "Inference Time",

    "Elapsed Time"

]

time_column = None

for col in possible_columns:

    if col in results.columns:

        time_column = col

        break

if time_column is None:

    raise ValueError(
        "Execution Time column not found."
    )

print(f"\nUsing time column : {time_column}")

# ==========================================================
# SUMMARY TABLE
# ==========================================================

summary = (

    results

    .groupby(

        ["Model", "Prompt"]

    )[time_column]

    .agg(

        Mean="mean",

        Median="median",

        Minimum="min",

        Maximum="max",

        Std="std"

    )

    .round(2)

    .reset_index()

)

summary.to_csv(

    os.path.join(

        TABLE_FOLDER,

        "execution_time_summary.csv"

    ),

    index=False

)

print()

print("=" * 60)
print("Execution Time Summary")
print("=" * 60)

print(summary)

# ==========================================================
# MODEL BOXPLOT
# ==========================================================

plt.figure(figsize=(8,6))

model_palette = {

    "gemini": "#1f77b4",

    "groq": "#2ca02c",

    "mistral": "#ff7f0e"

}

ax = sns.boxplot(

    data=results,

    x="Model",

    y=time_column,

    hue="Model",

    palette=model_palette

)

if ax.legend_:

    ax.legend_.remove()

plt.yscale("log")

plt.minorticks_off()

plt.title(

    "Execution Time by Model",

    fontsize=15,

    fontweight="bold"

)

plt.xlabel("Model")

plt.ylabel("Execution Time (seconds)")

plt.grid(

    axis="y",

    linestyle="--",

    alpha=0.3

)

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURE_FOLDER,

        "execution_time_by_model.png"

    ),

    dpi=300

)

plt.close()

# ==========================================================
# PROMPT BOXPLOT
# ==========================================================

plt.figure(figsize=(7,6))

prompt_palette = {

    "cot": "#4c72b0",

    "tot": "#dd8452"

}

ax = sns.boxplot(

    data=results,

    x="Prompt",

    y=time_column,

    hue="Prompt",

    palette=prompt_palette

)

if ax.legend_:

    ax.legend_.remove()

plt.xticks(

    [0,1],

    ["CoT","ToT"]

)

plt.yscale("log")

plt.minorticks_off()

plt.title(

    "Execution Time by Prompt",

    fontsize=15,

    fontweight="bold"

)

plt.xlabel("Prompt")

plt.ylabel("Execution Time (seconds)")

plt.grid(

    axis="y",

    linestyle="--",

    alpha=0.3

)

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURE_FOLDER,

        "execution_time_by_prompt.png"

    ),

    dpi=300

)

plt.close()

# ==========================================================
# AVERAGE TIME TABLES
# ==========================================================

model_average = (

    results

    .groupby("Model")[time_column]

    .mean()

    .round(2)

    .reset_index()

)

model_average.rename(

    columns={

        time_column: "Average Execution Time (s)"

    },

    inplace=True

)

model_average.to_csv(

    os.path.join(

        TABLE_FOLDER,

        "model_average_execution_time.csv"

    ),

    index=False

)

prompt_average = (

    results

    .groupby("Prompt")[time_column]

    .mean()

    .round(2)

    .reset_index()

)

prompt_average.rename(

    columns={

        time_column: "Average Execution Time (s)"

    },

    inplace=True

)

prompt_average.to_csv(

    os.path.join(

        TABLE_FOLDER,

        "prompt_average_execution_time.csv"

    ),

    index=False

)

print()

print("=" * 60)
print("Average Time by Model")
print("=" * 60)

print(model_average)

print()

print("=" * 60)
print("Average Time by Prompt")
print("=" * 60)

print(prompt_average)

# ==========================================================
# FINISHED
# ==========================================================

print()

print("=" * 60)
print("Figures Saved")
print("=" * 60)

print(

    os.path.join(

        FIGURE_FOLDER,

        "execution_time_by_model.png"

    )

)

print(

    os.path.join(

        FIGURE_FOLDER,

        "execution_time_by_prompt.png"

    )

)

print()

print("=" * 60)
print("Tables Saved")
print("=" * 60)

print(

    os.path.join(

        TABLE_FOLDER,

        "execution_time_summary.csv"

    )

)

print(

    os.path.join(

        TABLE_FOLDER,

        "model_average_execution_time.csv"

    )

)

print(

    os.path.join(

        TABLE_FOLDER,

        "prompt_average_execution_time.csv"

    )

)