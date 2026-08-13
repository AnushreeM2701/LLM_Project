"""
Step Count vs. correctness: summary table + per-question scatter, same
layout/pool as question_execution_time.py. Colour=Correctness, shape=Model.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from config.config import FIGURES_DIR, TABLES_DIR, MODEL_NAMES, PROMPT_TYPES, DIFFICULTIES
from src.utils.io import load_results
from src.analysis.question_execution_time import hard_tier_question_pool, MODEL_LABELS
from src.utils.stats import mannwhitney_test, logistic_regression

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
MODEL_SHAPES = {"gemini": "o", "groq": "s", "mistral": "^"}
YLIM = (0, 50)
OVERFLOW_MARKER_Y = 48.5


def step_count_summary() -> pd.DataFrame:
    """Model x Prompt x Difficulty x Correct/Incorrect -> N, mean, median step count."""

    df = load_results()
    df = df.copy()
    df["Answer Correct"] = df["Answer Correct"].astype(str).str.lower() == "true"
    df["Step Count"] = pd.to_numeric(df["Step Count"], errors="coerce")

    pool = hard_tier_question_pool(df)
    df = df[(df["Difficulty"] != "Hard") | (df["Question ID"].isin(pool))]

    rows = []
    for model in MODEL_NAMES:
        for prompt in PROMPT_TYPES:
            for difficulty in DIFFICULTIES:
                for correct in [True, False]:
                    subset = df[
                        (df["Model"] == model)
                        & (df["Prompt"] == prompt)
                        & (df["Difficulty"] == difficulty)
                        & (df["Answer Correct"] == correct)
                    ]
                    if len(subset) == 0:
                        continue
                    rows.append({
                        "Model": model,
                        "Prompt": prompt,
                        "Difficulty": difficulty,
                        "Correct": correct,
                        "N": len(subset),
                        "Mean Step Count": subset["Step Count"].mean(),
                        "Median Step Count": subset["Step Count"].median(),
                    })

    return pd.DataFrame(rows)


def step_count_significance() -> pd.DataFrame:
    """Model x Prompt x Difficulty: Mann-Whitney U test, correct vs. incorrect step counts."""

    df = load_results()
    df = df.copy()
    df["Answer Correct"] = df["Answer Correct"].astype(str).str.lower() == "true"
    df["Step Count"] = pd.to_numeric(df["Step Count"], errors="coerce")

    pool = hard_tier_question_pool(df)
    df = df[(df["Difficulty"] != "Hard") | (df["Question ID"].isin(pool))]

    rows = []
    for model in MODEL_NAMES:
        for prompt in PROMPT_TYPES:
            for difficulty in DIFFICULTIES:
                subset = df[
                    (df["Model"] == model)
                    & (df["Prompt"] == prompt)
                    & (df["Difficulty"] == difficulty)
                ]
                correct = subset[subset["Answer Correct"]]["Step Count"]
                incorrect = subset[~subset["Answer Correct"]]["Step Count"]

                result = mannwhitney_test(correct, incorrect)
                rows.append({
                    "Model": model,
                    "Prompt": prompt,
                    "Difficulty": difficulty,
                    "Mean Correct": correct.mean(),
                    "Median Correct": correct.median(),
                    "Mean Incorrect": incorrect.mean(),
                    "Median Incorrect": incorrect.median(),
                    **result,
                })

    return pd.DataFrame(rows)


def step_count_logistic_regression() -> pd.DataFrame:
    """Model x Prompt, Hard tier only: logistic regression of correctness on step count."""

    df = load_results()
    df = df.copy()
    df["Correct"] = (df["Answer Correct"].astype(str).str.lower() == "true").astype(int)
    df["StepCount"] = pd.to_numeric(df["Step Count"], errors="coerce")

    pool = hard_tier_question_pool(df)
    hard = df[(df["Difficulty"] == "Hard") & (df["Question ID"].isin(pool))]
    hard = hard.dropna(subset=["StepCount"])

    rows = []
    for model in MODEL_NAMES:
        for prompt in PROMPT_TYPES:
            subset = hard[(hard["Model"] == model) & (hard["Prompt"] == prompt)]

            result = logistic_regression(subset, "Correct ~ StepCount")

            rows.append({
                "Model": model,
                "Prompt": prompt,
                "Difficulty": "Hard",
                "N": result["n"],
                "StepCount Coef (log-odds)": result["coef"].get("StepCount"),
                "StepCount Odds Ratio": result["odds_ratio"].get("StepCount"),
                "StepCount p-value": result["p_value"].get("StepCount"),
                "StepCount Significant": result["p_value"].get("StepCount") < 0.05,
                "Pseudo R2": result["pseudo_r2"],
            })

    return pd.DataFrame(rows)


def plot_step_count_by_question(prompt: str, df: pd.DataFrame, question_pool: dict) -> None:
    """One panel per difficulty tier, colour=Correctness, shape=Model, shared Y-axis."""

    subset = df[df["Prompt"] == prompt].copy()
    subset["Answer Correct"] = subset["Answer Correct"].astype(str).str.lower() == "true"
    subset["Step Count"] = pd.to_numeric(subset["Step Count"], errors="coerce")

    fig, axes = plt.subplots(len(DIFFICULTIES), 1, figsize=(15, 3.6 * len(DIFFICULTIES)), sharex=False)

    for ax, difficulty in zip(axes, DIFFICULTIES):

        pool = question_pool[difficulty]
        diff_df = subset[(subset["Difficulty"] == difficulty) & (subset["Question ID"].isin(pool))].copy()

        position = {qid: i for i, qid in enumerate(pool)}
        diff_df["Tier Position"] = diff_df["Question ID"].map(position)

        for model in MODEL_NAMES:
            for correct, color, status in [(True, GOOD, "Correct"), (False, CRITICAL, "Incorrect")]:
                group = diff_df[(diff_df["Model"] == model) & (diff_df["Answer Correct"] == correct)]
                if len(group) == 0:
                    continue
                in_range = group[group["Step Count"] <= YLIM[1]]
                over = group[group["Step Count"] > YLIM[1]]

                ax.scatter(
                    in_range["Tier Position"], in_range["Step Count"],
                    label=f"{MODEL_LABELS[model]} - {status}", color=color, marker=MODEL_SHAPES[model],
                    s=45, alpha=0.85, edgecolor="black", linewidth=0.4,
                )
                if len(over):
                    # Distinct star marker, not reused by any model's shape.
                    ax.scatter(
                        over["Tier Position"], [OVERFLOW_MARKER_Y] * len(over),
                        marker="*", color=color, s=90, alpha=0.95,
                        edgecolor="black", linewidth=0.4, zorder=5,
                    )
                    for _, row in over.iterrows():
                        ax.annotate(
                            f"{model.capitalize()[:2]}:{row['Step Count']:.0f}",
                            (row["Tier Position"], OVERFLOW_MARKER_Y),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=5.5, fontweight="bold", color=color,
                        )

        ax.set_title(f"{difficulty} (n={len(pool)} questions)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Step Count")
        ax.set_ylim(YLIM)
        ax.set_xticks(range(len(pool)))
        ax.set_xticklabels(pool, rotation=90, fontsize=6)
        ax.set_xlabel("Question ID")
        ax.grid(linestyle="--", alpha=0.3)

    # De-duplicated legend: 3 shapes x 2 colours, built once.
    from matplotlib.lines import Line2D
    shape_handles = [
        Line2D([0], [0], marker=m, color="black", linestyle="", markerfacecolor="gray",
               markersize=8, label=MODEL_LABELS[model])
        for model, m in MODEL_SHAPES.items()
    ]
    color_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="", markerfacecolor=c,
               markersize=8, label=label)
        for c, label in [(GOOD, "Correct"), (CRITICAL, "Incorrect")]
    ]
    axes[0].legend(handles=shape_handles + color_handles, loc="upper left", ncol=2, fontsize=7)
    axes[0].text(
        1.0, 1.02, "* = actual value exceeds 50 (labelled Model:value)", transform=axes[0].transAxes,
        ha="right", va="bottom", fontsize=7, style="italic", color="#555",
    )

    fig.suptitle(
        f"Number of Steps Taken to Derive Each Answer ({prompt.upper()}) - "
        f"one panel per difficulty tier, shared Y-axis, shape=Model, colour=Correctness",
        fontsize=13, fontweight="bold")
    fig.text(
        0.5, -0.01,
        "mistral-large-latest resolved to Mistral Large 3 (mistral-large-2512) for these experiments (Aug 2026) — "
        "the alias is a moving target and would point to a newer model if re-run later.",
        ha="center", va="top", fontsize=6.5, style="italic", color="#666",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"step_count_by_question_{prompt}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")


def run():

    os.makedirs(TABLES_DIR, exist_ok=True)
    summary = step_count_summary()
    path = os.path.join(TABLES_DIR, "step_count_summary.csv")
    summary.to_csv(path, index=False)
    print(f"Saved -> {path}")
    print(summary)

    significance = step_count_significance()
    sig_path = os.path.join(TABLES_DIR, "rq2_step_count_mannwhitney.csv")
    significance.to_csv(sig_path, index=False)
    print(f"Saved -> {sig_path}")
    print(significance)

    logistic = step_count_logistic_regression()
    logistic_path = os.path.join(TABLES_DIR, "rq2_step_count_logistic_regression.csv")
    logistic.to_csv(logistic_path, index=False)
    print(f"Saved -> {logistic_path}")
    print(logistic)

    df = load_results()
    question_pool = {
        "Easy": sorted(df[df["Difficulty"] == "Easy"]["Question ID"].unique()),
        "Medium": sorted(df[df["Difficulty"] == "Medium"]["Question ID"].unique()),
        "Hard": hard_tier_question_pool(df),
    }

    for prompt in PROMPT_TYPES:
        plot_step_count_by_question(prompt, df, question_pool)


if __name__ == "__main__":
    run()
