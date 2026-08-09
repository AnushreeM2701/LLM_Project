"""
Word-frequency analysis of the free-text Error Subtype field, Hard tier,
restricted to the common-wrong-question intersection per prompt (equal N).
Hover meanings/examples are embedded as a legend panel in the image.
"""

import math
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from config.config import FIGURES_DIR, TABLES_DIR, MODEL_NAMES, PROMPT_TYPES
from src.utils.io import load_results
from src.analysis.error_definitions import WORD_DEFINITIONS

TOP_N = 10

# Stopwords + generic filler words that read as meaningless chart bars.
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "and", "or",
    "is", "was", "were", "be", "been", "as", "at", "by", "from", "that",
    "this", "it", "its", "than", "then", "but", "not", "no", "into",
    "instead", "when", "which", "their", "there", "using", "used", "use",
    "incorrect", "incorrectly",
    "all", "without", "must", "single", "final", "one", "some", "any",
    "each", "only", "same", "other", "another", "both",
}


def tokenize(text: str) -> list:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def common_wrong_hard_questions(df: pd.DataFrame, prompt: str) -> set:
    """Hard-tier Question IDs all 3 models got wrong, within one prompt."""

    sub = df[(df["Difficulty"] == "Hard") & (df["Prompt"] == prompt)]

    wrong_sets = []
    for model in MODEL_NAMES:
        wrong = sub[(sub["Model"] == model) & (~sub["Answer Correct"])]
        wrong_sets.append(set(wrong["Question ID"]))

    return set.intersection(*wrong_sets)


def word_frequency(subtypes: pd.Series) -> Counter:
    counter = Counter()
    for text in subtypes:
        counter.update(tokenize(str(text)))
    return counter


def word_example(subtypes: pd.Series, word: str) -> str:
    """First Error Subtype text whose tokens include this word."""

    for text in subtypes:
        if word in tokenize(str(text)):
            return str(text)
    return ""


def render_legend_panel(ax, words: list, examples: dict) -> None:
    ax.axis("off")
    ax.set_title("What each word means (hover content, shown here instead)", fontsize=10.5, fontweight="bold", loc="left")
    lines = []
    for w in words:
        meaning = WORD_DEFINITIONS.get(w, "No definition available.")
        example = examples.get(w)
        line = f"{w}  -  {meaning}"
        if example:
            line += f'\n    Example (this project): "{example}"'
        lines.append(line)
    mid = (len(lines) + 1) // 2
    ax.text(0.0, 0.95, "\n".join(lines[:mid]), transform=ax.transAxes, fontsize=8.5, ha="left", va="top")
    ax.text(0.52, 0.95, "\n".join(lines[mid:]), transform=ax.transAxes, fontsize=8.5, ha="left", va="top")


def run():

    df = load_results()
    df = df.copy()
    df["Answer Correct"] = df["Answer Correct"].astype(str).str.lower() == "true"

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for prompt in PROMPT_TYPES:

        common = common_wrong_hard_questions(df, prompt)
        pool = df[(df["Difficulty"] == "Hard") & (df["Prompt"] == prompt) & (df["Question ID"].isin(common))]

        # Shared top-10 words, fixed labels used by every model's chart.
        overall_counts = word_frequency(pool["Error Subtype"])
        top_words = [w for w, _ in overall_counts.most_common(TOP_N)]

        per_model_freq = {}
        shared_max = 0
        word_examples = {}
        for model in MODEL_NAMES:
            model_subtypes = pool[pool["Model"] == model]["Error Subtype"]
            counts = word_frequency(model_subtypes)
            freq = pd.DataFrame({"Word": top_words, "Count": [counts.get(w, 0) for w in top_words]})
            freq.to_csv(
                os.path.join(TABLES_DIR, f"error_subtype_word_frequency_hard_{model}_{prompt}.csv"),
                index=False,
            )
            print(f"Saved -> {os.path.join(TABLES_DIR, f'error_subtype_word_frequency_hard_{model}_{prompt}.csv')}")
            per_model_freq[model] = freq
            shared_max = max(shared_max, freq["Count"].max())
            for w in top_words:
                if w not in word_examples:
                    example = word_example(model_subtypes, w)
                    if example:
                        word_examples[w] = example

        xlim = (0, math.ceil(shared_max * 1.15) or 1)

        fig = plt.figure(figsize=(18, 9.5))
        gs = fig.add_gridspec(2, len(MODEL_NAMES), height_ratios=[3, 1.7], hspace=0.4, wspace=0.6)

        for col, model in enumerate(MODEL_NAMES):
            freq = per_model_freq[model]
            ordered = freq.sort_values("Count", ascending=True)
            ax = fig.add_subplot(gs[0, col])
            bars = ax.barh(ordered["Word"], ordered["Count"], color="#2a78d6", edgecolor="black", linewidth=0.5)
            for bar, count in zip(bars, ordered["Count"]):
                ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                         str(count), va="center", fontsize=8)
            ax.set_title(f"{model.capitalize()}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Frequency of Occurrence")
            ax.set_xlim(xlim)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        legend_ax = fig.add_subplot(gs[1, :])
        render_legend_panel(legend_ax, top_words, word_examples)

        fig.suptitle(
            f"Most Common Words in Error Descriptions (Hard Tier) ({prompt.upper()}) - "
            f"restricted to Hard-tier questions that ALL 3 models got wrong (n={len(common)})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        path = os.path.join(FIGURES_DIR, f"error_subtype_word_frequency_hard_by_model_{prompt}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {path}")


if __name__ == "__main__":
    run()
