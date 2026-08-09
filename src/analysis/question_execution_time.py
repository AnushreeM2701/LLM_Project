"""
Per-question execution time, one panel per difficulty tier, coloured by
model. Shared 0-300s Y-axis; overflow points drawn as a labelled triangle.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from config.config import FIGURES_DIR, MODEL_NAMES, PROMPT_TYPES, DIFFICULTIES, DATASET_SEED
from src.utils.io import load_results

# Fixed hue per model, used across every figure in this project.
MODEL_COLORS = {"gemini": "#2a78d6", "groq": "#eb6834", "mistral": "#1baf7a"}

# Exact model version actually served (Gemini's alias resolved to a
# different pinned version at run time; Groq/Mistral matched their config).
MODEL_LABELS = {
    "gemini": "Gemini (gemini-3.5-flash-lite)",
    "groq": "GPT-OSS-120B (Groq, openai/gpt-oss-120b)",
    "mistral": "Mistral Large (mistral-large-latest)",
}

QUESTION_POOL_SIZE = 40
YLIM = (0, 300)
OVERFLOW_MARKER_Y = 292


def hard_tier_question_pool(df: pd.DataFrame) -> list:
    """Seeded 40-of-51 Hard-tier sample, matching Easy/Medium's pool size."""

    hard_ids = sorted(df[df["Difficulty"] == "Hard"]["Question ID"].unique())
    sampled = pd.Series(hard_ids).sample(n=QUESTION_POOL_SIZE, random_state=DATASET_SEED)
    return sorted(sampled.tolist())


def plot_execution_time_by_difficulty(prompt: str, df: pd.DataFrame, question_pool: dict) -> None:
    """One panel per difficulty tier, points coloured by model, shared 0-300s Y-axis."""

    subset = df[df["Prompt"] == prompt].copy()
    subset["Execution Time (s)"] = pd.to_numeric(subset["Execution Time (s)"], errors="coerce")

    fig, axes = plt.subplots(len(DIFFICULTIES), 1, figsize=(15, 3.6 * len(DIFFICULTIES)), sharex=False)

    for ax, difficulty in zip(axes, DIFFICULTIES):

        pool = question_pool[difficulty]
        diff_df = subset[(subset["Difficulty"] == difficulty) & (subset["Question ID"].isin(pool))].copy()

        position = {qid: i for i, qid in enumerate(pool)}
        diff_df["Tier Position"] = diff_df["Question ID"].map(position)

        for model in MODEL_NAMES:
            model_df = diff_df[diff_df["Model"] == model]
            in_range = model_df[model_df["Execution Time (s)"] <= YLIM[1]]
            over = model_df[model_df["Execution Time (s)"] > YLIM[1]]

            ax.scatter(
                in_range["Tier Position"], in_range["Execution Time (s)"],
                label=MODEL_LABELS[model], color=MODEL_COLORS[model],
                s=45, alpha=0.85, edgecolor="black", linewidth=0.4,
            )
            if len(over):
                ax.scatter(
                    over["Tier Position"], [OVERFLOW_MARKER_Y] * len(over),
                    marker="^", color=MODEL_COLORS[model],
                    s=55, alpha=0.95, edgecolor="black", linewidth=0.4, zorder=5,
                )
                for _, row in over.iterrows():
                    ax.annotate(
                        f"{row['Execution Time (s)']:.0f}s",
                        (row["Tier Position"], OVERFLOW_MARKER_Y),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=5.5, fontweight="bold", color=MODEL_COLORS[model],
                    )

        ax.set_title(f"{difficulty} (n={len(pool)} questions)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Execution Time (s)")
        ax.set_ylim(YLIM)
        ax.set_xticks(range(len(pool)))
        ax.set_xticklabels(pool, rotation=90, fontsize=6)
        ax.set_xlabel("Question ID")
        ax.grid(linestyle="--", alpha=0.3)

    axes[0].legend(title="Model", loc="upper left", ncol=1, fontsize=7)
    axes[0].text(
        1.0, 1.02, "^ = actual value exceeds 300s (labelled)", transform=axes[0].transAxes,
        ha="right", va="bottom", fontsize=7, style="italic", color="#555",
    )

    fig.suptitle(f"Execution Time by Question, per Difficulty ({prompt.upper()})",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, -0.01,
        "mistral-large-latest resolved to Mistral Large 3 (mistral-large-2512) for these experiments (Aug 2026) — "
        "the alias is a moving target and would point to a newer model if re-run later.",
        ha="center", va="top", fontsize=6.5, style="italic", color="#666",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"execution_time_by_difficulty_{prompt}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")


def run():
    df = load_results()

    question_pool = {
        "Easy": sorted(df[df["Difficulty"] == "Easy"]["Question ID"].unique()),
        "Medium": sorted(df[df["Difficulty"] == "Medium"]["Question ID"].unique()),
        "Hard": hard_tier_question_pool(df),
    }

    for prompt in PROMPT_TYPES:
        plot_execution_time_by_difficulty(prompt, df, question_pool)


if __name__ == "__main__":
    run()
