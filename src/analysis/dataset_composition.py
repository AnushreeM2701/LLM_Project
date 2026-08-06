"""
Dataset composition chart: one wedge per category, sized proportionally to
its actual question count (NOT equal wedges) -- the Hard tier is
deliberately unbalanced across category (see docs/methodology.md: Probability
tops out at 8 curated AIME questions vs. 16 for Algebra/Combinatorics), so an
artificially even-quartered chart would misrepresent the real composition.

Each wedge is also labelled with its exact Total and Easy/Medium/Hard
counts, so the reader isn't left eyeballing slice angles to compare
categories whose totals are fairly close (36/31/28/36) -- per the general
guidance that pie/donut charts should carry their numbers directly when
segment sizes are close enough to be hard to compare by eye alone.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from config.config import FIGURES_DIR, CATEGORIES, DIFFICULTIES
from src.utils.io import load_dataset

# Categorical palette, fixed order, one hue per category (never cycled).
CATEGORY_COLORS = {
    "Algebra": "#2a78d6",
    "Number Theory": "#eb6834",
    "Probability": "#1baf7a",
    "Combinatorics": "#eda100",
}


def composition_table():

    df = load_dataset()

    table = {}
    for category in CATEGORIES:
        cat_df = df[df["Category"] == category]
        counts = {d: int((cat_df["Difficulty"] == d).sum()) for d in DIFFICULTIES}
        counts["Total"] = len(cat_df)
        table[category] = counts

    return table


def plot_composition_pie() -> None:

    table = composition_table()
    totals = [table[c]["Total"] for c in CATEGORIES]
    colors = [CATEGORY_COLORS[c] for c in CATEGORIES]

    fig, ax = plt.subplots(figsize=(9, 9))

    wedges, _ = ax.pie(
        totals,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "black", "linewidth": 1.5},
    )

    for wedge, category in zip(wedges, CATEGORIES):
        angle = (wedge.theta2 + wedge.theta1) / 2
        rad = np.deg2rad(angle)
        label_x = 0.62 * wedge.r * np.cos(rad)
        label_y = 0.62 * wedge.r * np.sin(rad)

        counts = table[category]
        text = (
            f"{category}\n\n"
            f"Total: {counts['Total']}\n\n"
            f"Easy: {counts['Easy']}\n"
            f"Medium: {counts['Medium']}\n"
            f"Hard: {counts['Hard']}"
        )
        ax.text(label_x, label_y, text, ha="center", va="center",
                 fontsize=11, fontweight="bold", color="black")

    ax.set_title("Dataset Composition by Category", fontsize=13, fontweight="bold")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "dataset_composition_pie.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {path}")
    for category in CATEGORIES:
        print(category, table[category])


def run():
    plot_composition_pie()


if __name__ == "__main__":
    run()
