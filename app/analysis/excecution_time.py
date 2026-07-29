import os
import pandas as pd
import matplotlib.pyplot as plt
import re
# ==========================================================
# FILE PATH
# ==========================================================

RESULTS_FILE = "data/results/experiment_results.csv"

OUTPUT_FOLDER = "outputs"
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "figures")

os.makedirs(FIGURE_FOLDER, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

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
# CREATE PLOT QUESTION (1-40)
# ==========================================================

category_order = {
    "ALG": 0,
    "NT": 1,
    "PROB": 2,
    "COMB": 3
}

results["CategoryPrefix"] = results["Question ID"].str.split("_").str[0]

results["QuestionInCategory"] = (
    results["Question ID"]
    .str.extract(r'_(\d+)$')
    .astype(int)
)

results["Plot Question"] = (
    results["CategoryPrefix"].map(category_order) * 10
    + results["QuestionInCategory"]
)
print(
    results.groupby(["Category", "Difficulty"])["Question ID"].nunique()
)
# ==========================================================
# COLORS
# ==========================================================

colors = {
    "gemini": "#1f77b4",
    "groq": "#ff7f0e",
    "mistral": "#2ca02c"
}

# ==========================================================
# COMMON Y LIMIT
# ==========================================================

ymax = results[time_column].max() * 1.05

# ==========================================================
# CREATE CoT AND ToT
# ==========================================================

for prompt in ["cot", "tot"]:

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 12),
        sharex=True,
        sharey=False
    )

    prompt_df = results[
        results["Prompt"].str.lower() == prompt
    ]

    difficulties = ["Easy", "Medium", "Hard"]

    for ax, difficulty in zip(axes, difficulties):

        df_diff = prompt_df[
            prompt_df["Difficulty"] == difficulty
        ]

        for model in ["gemini", "groq", "mistral"]:

            df = df_diff[
                df_diff["Model"].str.lower() == model
            ]

            ax.scatter(
                df["Plot Question"],
                df[time_column],
                color=colors[model],
                marker="o",
                s=45,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.4,
                label=model.capitalize()
            )

        ax.set_title(
            difficulty,
            fontsize=12,
            fontweight="bold"
        )

        ax.set_ylabel(
            "Execution Time (s)",
            fontsize=10
        )

        if difficulty == "Easy":
            ax.set_ylim(0, 20)
            ax.set_yticks(range(0, 20, 2))

        elif difficulty == "Medium":
            ax.set_ylim(0, 20)
            ax.set_yticks(range(0, 21, 2))

        else:   # Hard
            ax.set_ylim(0, 80)
            ax.set_yticks(range(0, 80, 10))

        ax.set_xlim(1, 40)

        ax.set_xticks(range(1, 41, 2))

        ax.grid(
            linestyle="--",
            alpha=0.3
        )

    axes[-1].set_xlabel(
        "Question Number (1–40)",
        fontsize=11,
        fontweight="bold"
    )

    handles, labels = axes[0].get_legend_handles_labels()

    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=3,
        title="Model",
        fontsize=10
    )

    fig.suptitle(
        f"Execution Time ({prompt.upper()})",
        fontsize=16,
        fontweight="bold",
        y=0.94
    )

    fig.subplots_adjust(
        top=0.90,
        hspace=0.35
    )

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            f"execution_time_{prompt}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Done!")
print("Saved:")
print("execution_time_cot.png")
print("execution_time_tot.png")