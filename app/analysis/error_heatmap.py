import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import FancyBboxPatch

# ==========================================================
# FILE PATHS
# ==========================================================

FILE_PATH = "data/results/experiment_results.xlsx"
SHEET_NAME = "Experiment Results"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("Loading Experiment Results...")
print("=" * 60)

df = pd.read_excel(
    FILE_PATH,
    sheet_name=SHEET_NAME
)

# ==========================================================
# CLEAN DATA
# ==========================================================

df.columns = df.columns.str.strip()

df["Model"] = (
    df["Model"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["Prompt"] = (
    df["Prompt"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["Difficulty"] = (
    df["Difficulty"]
    .astype(str)
    .str.strip()
    .str.title()
)

df["Error Type"] = (
    df["Error Type"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# ==========================================================
# REMOVE NON-ERRORS
# ==========================================================

df = df[
    df["Error Type"] != "Correct"
].copy()

# ==========================================================
# KEEP ONLY TOP 6 ERROR TYPES
# ==========================================================

TOP_ERRORS = [

    "Combinatorial Counting Error",

    "Logical Reasoning Error",

    "Algebraic Manipulation Error",

    "Incorrect Assumption",

    "Number Theory Error",

    "Arithmetic Error"

]

df = df[
    df["Error Type"].isin(TOP_ERRORS)
].copy()

# ==========================================================
# ORDERING
# ==========================================================

MODELS = [

    "gemini",

    "groq",

    "mistral"

]

DIFFICULTIES = [

    "Easy",

    "Medium",

    "Hard"

]

# ==========================================================
# COLOR MAP
# Similar to your reference image
# ==========================================================

colors = [

    "#eef5fc",

    "#d8e7f7",

    "#b9d2ec",

    "#88add1",

    "#4d78aa",

    "#1f4f85"

]

cmap = mcolors.LinearSegmentedColormap.from_list(
    "paper_blue",
    colors
)

print()
print("Data Loaded Successfully")
print(f"Rows after filtering : {len(df)}")
print()

# ==========================================================
# CREATE PIVOT TABLE
# ==========================================================

def create_matrix(prompt):

    temp = df[
        df["Prompt"] == prompt.lower()
    ].copy()

    rows = []

    index = []

    for model in MODELS:

        for difficulty in DIFFICULTIES:

            subset = temp[
                (temp["Model"] == model) &
                (temp["Difficulty"] == difficulty)
            ]

            counts = []

            for error in TOP_ERRORS:

                counts.append(
                    len(
                        subset[
                            subset["Error Type"] == error
                        ]
                    )
                )

            rows.append(counts)

            index.append(
                (
                    model.capitalize(),
                    difficulty
                )
            )

    matrix = pd.DataFrame(
        rows,
        columns=TOP_ERRORS,
        index=pd.MultiIndex.from_tuples(
            index,
            names=["Model", "Difficulty"]
        )
    )

    return matrix


# ==========================================================
# DRAW HEATMAP
# ==========================================================

def draw_heatmap(matrix, title, save_path):

    nrows = matrix.shape[0]
    ncols = matrix.shape[1]

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    ax.invert_yaxis()

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    vmax = matrix.to_numpy().max()

    if vmax == 0:
        vmax = 1

    norm = mcolors.Normalize(
        vmin=0,
        vmax=vmax
    )

    # =====================================================
    # Draw rounded cells
    # =====================================================

    for i in range(nrows):

        for j in range(ncols):

            value = matrix.iloc[i, j]

            color = cmap(norm(value))

            cell = FancyBboxPatch(

                (j + 0.05, i + 0.05),

                0.90,

                0.90,

                boxstyle="round,pad=0.02",

                linewidth=1,

                edgecolor="white",

                facecolor=color

            )

            ax.add_patch(cell)

            ax.text(

                j + 0.5,

                i + 0.5,

                str(value),

                ha="center",

                va="center",

                fontsize=11,

                fontweight="bold",

                color="black"

            )

    # =====================================================
    # Axis Labels
    # =====================================================

    ax.set_xticks(
        np.arange(ncols) + 0.5
    )

    ax.set_xticklabels(

        [
            "Counting",
            "Logical",
            "Algebra",
            "Assumption",
            "Number\nTheory",
            "Arithmetic"
        ],

        fontsize=10,

        fontweight="bold"

    )

    ylabels = []

    for idx, (model, difficulty) in enumerate(matrix.index):

        if idx % 3 == 0:
            ylabels.append(f"{model}\n{difficulty}")
        else:
            ylabels.append(difficulty)

    ax.set_yticks(np.arange(nrows) + 0.5)

    ax.set_yticklabels(
        [
            "Easy",
            "Medium",
            "Hard",
            "Easy",
            "Medium",
            "Hard",
            "Easy",
            "Medium",
            "Hard",
        ],
        fontsize=10
    )

    # Model labels (display only once)
    ax.text(
        -1.30, 1.5, "Gemini",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center"
    )

    ax.text(
        -1.30, 4.5, "Groq",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center"
    )

    ax.text(
        -1.30, 7.5, "Mistral",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center"
    )

    # =====================================================
    # Group separator
    # =====================================================

    for y in [3, 6]:

        ax.axhline(
            y,
            color="gray",
            linewidth=2
        )

    # =====================================================
    # Remove ticks
    # =====================================================

    ax.tick_params(length=0)

    for spine in ax.spines.values():

        spine.set_visible(False)

    plt.title(

        title,

        fontsize=18,

        fontweight="bold",

        pad=20

    )

    # =====================================================
    # Colorbar
    # =====================================================

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )

    sm.set_array([])

    plt.colorbar(
        sm,
        ax=ax,
        fraction=0.03,
        pad=0.02
    )

    plt.subplots_adjust(left=0.28)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved -> {save_path}")

    # ==========================================================
# GENERATE CoT HEATMAP
# ==========================================================

cot_matrix = create_matrix("cot")

draw_heatmap(
    matrix=cot_matrix,
    title="CoT Error Distribution by Model and Difficulty",
    save_path=os.path.join(
        OUTPUT_DIR,
        "cot_error_heatmap.png"
    )
)

# ==========================================================
# GENERATE ToT HEATMAP
# ==========================================================

tot_matrix = create_matrix("tot")

draw_heatmap(
    matrix=tot_matrix,
    title="ToT Error Distribution by Model and Difficulty",
    save_path=os.path.join(
        OUTPUT_DIR,
        "tot_error_heatmap.png"
    )
)

# ==========================================================
# PRINT MATRICES (Optional)
# ==========================================================

print("\n================ CoT Matrix ================\n")
print(cot_matrix)

print("\n================ ToT Matrix ================\n")
print(tot_matrix)

print("\n============================================")
print("Heatmaps generated successfully!")
print(f"Saved to: {OUTPUT_DIR}")
print("============================================")


