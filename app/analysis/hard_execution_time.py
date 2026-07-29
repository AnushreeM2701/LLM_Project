import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# FILE PATHS
# ==========================================================

RESULTS_FILE = "data/results/experiment_results copy.csv"

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
# LOAD DATA
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

    "Inference Time"

]

time_column = None

for col in possible_columns:

    if col in results.columns:

        time_column = col

        break

if time_column is None:

    raise ValueError("Execution Time column not found.")

# ==========================================================
# FILTER HARD QUESTIONS
# ==========================================================

PROMPT = "cot"          # Change to "tot" if required

results = results[
    (results["Difficulty"] == "Hard") &
    (results["Prompt"] == PROMPT)
].copy()

results = results.sort_values(
    "Question Number"
)

# ==========================================================
# SAVE TABLE
# ==========================================================

table_path = os.path.join(
    TABLE_FOLDER,
    f"hard_execution_time_{PROMPT}.csv"
)

results.to_csv(
    table_path,
    index=False
)

results.to_excel(
    table_path.replace(".csv", ".xlsx"),
    index=False
)

# ==========================================================
# SCATTER PLOT
# ==========================================================

plt.figure(figsize=(12,5))

markers = {

    "gemini": "o",

    "groq": "^",

    "mistral": "s"

}

for model in ["gemini", "groq", "mistral"]:

    df = results[
        results["Model"] == model
    ]

    plt.scatter(

        df["Question Number"],

        df[time_column],

        marker=markers[model],

        s=70,

        label=model.capitalize()

    )

# ==========================================================
# FORMATTING
# ==========================================================

plt.xlabel(
    "Hard Question Number"
)

plt.ylabel(
    "Execution Time (seconds)"
)

plt.title(
    f"Execution Time for Hard Questions ({PROMPT.upper()})",
    fontsize=14,
    fontweight="bold"
)

plt.grid(
    linestyle="--",
    alpha=0.3
)

plt.legend()

plt.tight_layout()

figure_path = os.path.join(

    FIGURE_FOLDER,

    f"hard_execution_time_{PROMPT}.png"

)

plt.savefig(

    figure_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print()
print("=" * 60)
print("Figure Saved")
print("=" * 60)

print(figure_path)