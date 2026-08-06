"""
Word-frequency analysis of the free-text Error Subtype field -- surfaces
recurring failure themes underneath the 11 fixed Error Type categories
(e.g. how often "counted"/"assumed"/"ignored" language shows up).

Restricted to the Hard tier: 124 of 131 non-empty subtypes are Hard-tier
already (Easy has 6, Medium has 1), so this is close to the full picture
anyway, and it keeps the comparison to a single, consistent difficulty
level rather than mixing in a handful of Easy/Medium rows.

"incorrect"/"incorrectly" are excluded from the word list -- they're
near-universal in this context (every subtype describes an incorrect
action by definition), so they don't distinguish anything; words like
"assumed"/"counted"/"guessed"/"failed" describe the SPECIFIC nature of the
mistake and are the actual signal.
"""

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from config.config import FIGURES_DIR, TABLES_DIR, MODEL_NAMES
from src.utils.io import load_results

TOP_N = 15

# True stopwords only (articles, prepositions, pronouns, auxiliaries) --
# error-describing verbs like "counted"/"assumed"/"ignored" are kept
# deliberately, since those are exactly the recurring signal this analysis
# is looking for.
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "and", "or",
    "is", "was", "were", "be", "been", "as", "at", "by", "from", "that",
    "this", "it", "its", "than", "then", "but", "not", "no", "into",
    "instead", "when", "which", "their", "there", "using", "used", "use",
    "incorrect", "incorrectly",
}


def tokenize(text: str) -> list:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def word_frequency(subtypes: pd.Series, top_n: int = TOP_N) -> pd.DataFrame:

    counter = Counter()
    for text in subtypes:
        counter.update(tokenize(text))

    freq = pd.DataFrame(counter.most_common(top_n), columns=["Word", "Count"])
    return freq


def plot_word_frequency(freq: pd.DataFrame, title: str, path: str) -> None:

    ordered = freq.sort_values("Count", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(ordered["Word"], ordered["Count"], color="#2a78d6", edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, ordered["Count"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 str(count), va="center", fontsize=9)

    ax.set_xlabel("Occurrences")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")


def run():

    df = load_results()
    subtypes = df[
        (df["Error Subtype"].astype(str).str.strip() != "")
        & (df["Difficulty"] == "Hard")
    ]

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Overall (Hard tier)
    overall_freq = word_frequency(subtypes["Error Subtype"])
    overall_freq.to_csv(os.path.join(TABLES_DIR, "error_subtype_word_frequency_hard.csv"), index=False)
    print(f"Saved -> {os.path.join(TABLES_DIR, 'error_subtype_word_frequency_hard.csv')}")
    print(overall_freq)
    plot_word_frequency(
        overall_freq,
        f"Most Common Words in Error Subtypes, Hard Tier (n={len(subtypes)} incorrect responses)",
        os.path.join(FIGURES_DIR, "error_subtype_word_frequency_hard.png"),
    )

    # Model-wise
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(16, 6))
    for ax, model in zip(axes, MODEL_NAMES):
        model_subtypes = subtypes[subtypes["Model"] == model]["Error Subtype"]
        freq = word_frequency(model_subtypes, top_n=10)
        freq.to_csv(os.path.join(TABLES_DIR, f"error_subtype_word_frequency_hard_{model}.csv"), index=False)
        print(f"Saved -> {os.path.join(TABLES_DIR, f'error_subtype_word_frequency_hard_{model}.csv')}")

        ordered = freq.sort_values("Count", ascending=True)
        bars = ax.barh(ordered["Word"], ordered["Count"], color="#2a78d6", edgecolor="black", linewidth=0.5)
        for bar, count in zip(bars, ordered["Count"]):
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                     str(count), va="center", fontsize=8)
        ax.set_title(f"{model.capitalize()} (n={len(model_subtypes)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Occurrences")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Most Common Words in Error Subtypes, Hard Tier, by Model", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(FIGURES_DIR, "error_subtype_word_frequency_hard_by_model.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


if __name__ == "__main__":
    run()
