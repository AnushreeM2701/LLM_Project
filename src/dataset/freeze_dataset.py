"""
Builds the frozen final_dataset.xlsx used by every downstream experiment.
"""

import os
import textwrap

import pandas as pd
from openpyxl.styles import Alignment

from config.config import (
    AIME_MASTER_PATH,
    HENDRYCKS_CANDIDATES_PATH,
    FINAL_DATASET_PATH,
    METHODOLOGY_DOC_PATH,
    CATEGORIES,
    DATASET_SEED,
    EASY_MEDIUM_PER_CATEGORY,
)

CATEGORY_PREFIX = {
    "Algebra": "ALG",
    "Number Theory": "NT",
    "Probability": "PROB",
    "Combinatorics": "COMB",
}

DIFFICULTY_SUFFIX = {"Easy": "E", "Medium": "M", "Hard": "H"}

AUTO_BLOCK_START = "<!-- BEGIN AUTO-GENERATED COMPOSITION TABLE (written by src/dataset/freeze_dataset.py — do not hand-edit) -->"
AUTO_BLOCK_END = "<!-- END AUTO-GENERATED COMPOSITION TABLE -->"


# ==========================================================
# LOAD SOURCES
# ==========================================================

def load_aime_hard() -> pd.DataFrame:

    aime = pd.read_excel(AIME_MASTER_PATH)

    aime = aime[aime["Include"] == "T"].copy()
    aime = aime[aime["Category"].isin(CATEGORIES)].copy()  # drops Geometry (out of study scope)

    aime = aime.rename(columns={"Official Answer": "Ground Truth Final Answer"})
    aime["Source"] = "AIME"
    aime["Difficulty"] = "Hard"

    if "Ground Truth Solution" not in aime.columns:
        aime["Ground Truth Solution"] = ""

    return aime


def load_hendrycks_easy_medium() -> pd.DataFrame:

    xls = pd.ExcelFile(HENDRYCKS_CANDIDATES_PATH)

    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(HENDRYCKS_CANDIDATES_PATH, sheet_name=sheet)
        df = df[df["Include"] == "T"].copy()
        df["Source"] = "Hendrycks"
        frames.append(df)

    hendrycks = pd.concat(frames, ignore_index=True, sort=False)

    hendrycks["Category"] = hendrycks.apply(
        lambda row: row["Final Category"]
        if pd.notna(row.get("Final Category")) and str(row["Final Category"]).strip() != ""
        else row["Category"],
        axis=1,
    )

    # Hard tier is AIME-only in this pipeline — any Hendrycks row still
    # tagged Hard/Level 5 in the curated file is dropped here.
    hendrycks = hendrycks[hendrycks["Difficulty"] != "Hard"].copy()
    hendrycks = hendrycks[hendrycks["Category"].isin(CATEGORIES)].copy()

    return hendrycks


def balance_easy_medium(df: pd.DataFrame) -> pd.DataFrame:

    balanced = []

    for category in CATEGORIES:
        for difficulty in ["Easy", "Medium"]:

            subset = df[(df["Category"] == category) & (df["Difficulty"] == difficulty)]

            if len(subset) > EASY_MEDIUM_PER_CATEGORY:
                subset = subset.sample(n=EASY_MEDIUM_PER_CATEGORY, random_state=DATASET_SEED)

            balanced.append(subset)

    return pd.concat(balanced, ignore_index=True)


# ==========================================================
# BUILD
# ==========================================================

def build_dataset() -> pd.DataFrame:

    aime_hard = load_aime_hard()
    hendrycks = load_hendrycks_easy_medium()
    easy_medium = balance_easy_medium(hendrycks)

    dataset = pd.concat([easy_medium, aime_hard], ignore_index=True, sort=False)

    columns = [
        "Source", "Category", "Difficulty", "Question",
        "Ground Truth Final Answer", "Ground Truth Solution",
    ]
    for col in columns:
        if col not in dataset.columns:
            dataset[col] = ""
    dataset = dataset[columns]

    dataset["Category"] = pd.Categorical(dataset["Category"], categories=CATEGORIES, ordered=True)
    dataset["Difficulty"] = pd.Categorical(dataset["Difficulty"], categories=["Easy", "Medium", "Hard"], ordered=True)
    dataset = dataset.sort_values(["Category", "Difficulty"]).reset_index(drop=True)

    # Question IDs
    counter = {}
    question_ids = []
    for _, row in dataset.iterrows():
        key = (row["Category"], row["Difficulty"])
        counter[key] = counter.get(key, 0) + 1
        qid = f"{CATEGORY_PREFIX[row['Category']]}_{DIFFICULTY_SUFFIX[row['Difficulty']]}_{counter[key]:03}"
        question_ids.append(qid)
    dataset.insert(0, "Question ID", question_ids)

    dataset.insert(0, "Question Number", range(1, len(dataset) + 1))

    dataset["Question"] = dataset["Question"].fillna("").apply(
        lambda x: "\n".join(textwrap.wrap(str(x), width=70))
    )
    dataset["Ground Truth Solution"] = dataset["Ground Truth Solution"].fillna("").apply(
        lambda x: "\n".join(textwrap.wrap(str(x), width=90))
    )
    dataset["Ground Truth Final Answer"] = dataset["Ground Truth Final Answer"].fillna("").astype(str)

    return dataset


def save_dataset(dataset: pd.DataFrame) -> None:

    os.makedirs(os.path.dirname(FINAL_DATASET_PATH), exist_ok=True)

    with pd.ExcelWriter(FINAL_DATASET_PATH, engine="openpyxl") as writer:

        dataset.to_excel(writer, index=False, sheet_name="Final Dataset")
        worksheet = writer.sheets["Final Dataset"]

        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        widths = {"A": 15, "B": 18, "C": 15, "D": 20, "E": 12, "F": 70, "G": 20, "H": 90}
        for col, width in widths.items():
            worksheet.column_dimensions[col].width = width

        for row in worksheet.iter_rows():
            max_lines = 1
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                if cell.value:
                    max_lines = max(max_lines, str(cell.value).count("\n") + 1)
            worksheet.row_dimensions[row[0].row].height = max(25, max_lines * 18)


# ==========================================================
# COMPOSITION TABLE -> docs/methodology.md
# ==========================================================

def composition_table_markdown(dataset: pd.DataFrame) -> str:

    lines = [AUTO_BLOCK_START, ""]
    lines.append("### Dataset Composition")
    lines.append("")
    lines.append(f"Total questions: **{len(dataset)}**")
    lines.append("")
    lines.append("| Category | Easy | Medium | Hard (AIME) | Total |")
    lines.append("|---|---|---|---|---|")

    for category in CATEGORIES:
        cat_df = dataset[dataset["Category"] == category]
        easy = (cat_df["Difficulty"] == "Easy").sum()
        medium = (cat_df["Difficulty"] == "Medium").sum()
        hard = (cat_df["Difficulty"] == "Hard").sum()
        lines.append(f"| {category} | {easy} | {medium} | {hard} | {easy + medium + hard} |")

    total_hard = (dataset["Difficulty"] == "Hard").sum()
    lines.append(f"| **Total** | {(dataset['Difficulty']=='Easy').sum()} | {(dataset['Difficulty']=='Medium').sum()} | {total_hard} | {len(dataset)} |")

    lines.append("")
    lines.append(
        "Hard tier is **AIME-only** (Hendrycks Level-5 excluded — see "
        "`docs/limitations.md` for the rationale). Category "
        "counts within Hard are **unbalanced by design**: Probability caps "
        "out at 8 available curated AIME questions, and the other "
        "categories are not trimmed to match, to preserve statistical "
        "power. Category is treated as a covariate in analysis rather "
        "than requiring equal N."
    )
    lines.append("")
    lines.append(f"Easy/Medium tiers are Hendrycks MATH, balanced to "
                  f"{EASY_MEDIUM_PER_CATEGORY} questions per category "
                  f"(seed={DATASET_SEED}).")
    lines.append("")
    lines.append(AUTO_BLOCK_END)

    return "\n".join(lines)


def update_methodology_doc(dataset: pd.DataFrame) -> None:

    table_md = composition_table_markdown(dataset)

    if os.path.exists(METHODOLOGY_DOC_PATH):
        with open(METHODOLOGY_DOC_PATH, "r") as f:
            content = f.read()
    else:
        content = "# Dataset Methodology\n\n"

    if AUTO_BLOCK_START in content and AUTO_BLOCK_END in content:
        before = content.split(AUTO_BLOCK_START)[0]
        after = content.split(AUTO_BLOCK_END)[1]
        content = before + table_md + after
    else:
        content = content.rstrip() + "\n\n" + table_md + "\n"

    with open(METHODOLOGY_DOC_PATH, "w") as f:
        f.write(content)


# ==========================================================
# MAIN
# ==========================================================

def freeze_dataset():

    dataset = build_dataset()
    save_dataset(dataset)
    update_methodology_doc(dataset)

    print("=" * 60)
    print("Final Dataset Frozen")
    print(FINAL_DATASET_PATH)
    print("=" * 60)
    print()
    print(dataset.groupby(["Category", "Difficulty"], observed=True).size())
    print()
    print(f"Methodology composition table written -> {METHODOLOGY_DOC_PATH}")

    return dataset


if __name__ == "__main__":
    freeze_dataset()
