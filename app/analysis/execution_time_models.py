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
# COLORS (Difficulty)
# ==========================================================

colors = {
    "Easy": "#2ca02c",      # Green
    "Medium": "#ff7f0e",    # Orange
    "Hard": "#d62728"       # Red
}

# ==========================================================
# CREATE CoT AND ToT
# ==========================================================

for prompt in ["cot", "tot"]:

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 12),
        sharex=True
    )

    prompt_df = results[
        results["Prompt"].str.lower() == prompt
    ]

    models = ["gemini", "groq", "mistral"]

    for ax, model in zip(axes, models):

        model_df = prompt_df[
            prompt_df["Model"].str.lower() == model
        ]

        for difficulty in ["Easy", "Medium", "Hard"]:

            df = model_df[
                model_df["Difficulty"] == difficulty
            ].sort_values("Plot Question")

            ax.scatter(
                df["Plot Question"],
                df[time_column],
                color=colors[difficulty],
                marker="o",
                s=45,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.4,
                label=difficulty
            )

        ax.set_title(
            model.capitalize(),
            fontsize=12,
            fontweight="bold"
        )

        ax.set_ylabel(
            "Execution Time (s)",
            fontsize=10
        )
        if model == "gemini":
            ax.set_ylim(0, 15)
            ax.set_yticks(range(0, 16, 3))

        elif model == "groq":
            ax.set_ylim(0, 25)
            ax.set_yticks(range(0, 26, 2))

        elif model == "mistral":
            ax.set_ylim(0, 30)
            ax.set_yticks(range(0, 31, 3))

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
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        title="Difficulty",
        fontsize=10
    )

    fig.suptitle(
        f"Execution Time ({prompt.upper()})",
        fontsize=16,
        fontweight="bold",
        y=0.94
    )

    fig.subplots_adjust(
        top=0.88,
        hspace=0.35
    )

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            f"execution_time_by_model_{prompt}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Done!")
print("Saved:")
print("execution_time_by_model_cot.png")
print("execution_time_by_model_tot.png")