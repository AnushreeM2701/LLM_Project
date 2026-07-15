import os
import textwrap

import pandas as pd
from openpyxl.styles import Alignment

# ==========================================================
# FILE PATHS
# ==========================================================

AIME_FILE = "data/raw/master_dataset.xlsx"

EASY_MEDIUM_FILE = "data/candidates/easy_medium_candidates.xlsx"

OUTPUT_FOLDER = "data/processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "final_dataset.xlsx"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LOAD AIME
# ==========================================================

print("Loading AIME dataset...")

aime = pd.read_excel(AIME_FILE)

aime.rename(
    columns={
        "Official Answer": "Ground Truth Final Answer"
    },
    inplace=True
)

aime = aime[aime["Include"] == "T"].copy()

aime["Source"] = "AIME"

if "Ground Truth Solution" not in aime.columns:
    aime["Ground Truth Solution"] = ""

# ==========================================================
# LOAD EASY / MEDIUM
# ==========================================================

print("Loading Easy/Medium dataset...")

xls = pd.ExcelFile(EASY_MEDIUM_FILE)

frames = []

for sheet in xls.sheet_names:

    df = pd.read_excel(EASY_MEDIUM_FILE, sheet_name=sheet)

    df = df[df["Include"] == "T"].copy()

    df["Source"] = "Hendrycks"

    frames.append(df)

easy_medium = pd.concat(
    frames,
    ignore_index=True
)

# ==========================================================
# MERGE
# ==========================================================

dataset = pd.concat(
    [easy_medium, aime],
    ignore_index=True,
    sort=False
).reset_index(drop=True)

# ==========================================================
# USE FINAL CATEGORY
# ==========================================================

dataset["Category"] = dataset.apply(
    lambda row:
    row["Final Category"]
    if pd.notna(row.get("Final Category"))
    and str(row["Final Category"]).strip() != ""
    else row["Category"],
    axis=1
)

# ==========================================================
# GENERATE QUESTION IDs
# ==========================================================

prefix = {
    "Algebra": "ALG",
    "Number Theory": "NT",
    "Probability": "PROB",
    "Combinatorics": "COMB"
}

difficulty = {
    "Easy": "E",
    "Medium": "M",
    "Hard": "H"
}

counter = {}

question_ids = []

for _, row in dataset.iterrows():

    key = (row["Category"], row["Difficulty"])

    counter[key] = counter.get(key, 0) + 1

    qid = (
        f"{prefix[row['Category']]}_"
        f"{difficulty[row['Difficulty']]}_"
        f"{counter[key]:03}"
    )

    question_ids.append(qid)

if "Question ID" in dataset.columns:
    dataset.drop(columns=["Question ID"], inplace=True)

dataset.insert(0, "Question ID", question_ids)

# ==========================================================
# REMOVE UNNECESSARY COLUMNS
# ==========================================================

dataset.drop(
    columns=[
        "Include",
        "Final Category",
        "Original Subject",
        "Original Level"
    ],
    inplace=True,
    errors="ignore"
)

# ==========================================================
# COLUMN ORDER
# ==========================================================

columns = [

    "Question ID",

    "Source",

    "Category",

    "Difficulty",

    "Question",

    "Ground Truth Final Answer",

    "Ground Truth Solution"

]

for col in columns:

    if col not in dataset.columns:

        dataset[col] = ""

dataset = dataset[columns]

# ==========================================================
# SORT
# ==========================================================

dataset["Category"] = pd.Categorical(

    dataset["Category"],

    categories=[
        "Algebra",
        "Number Theory",
        "Probability",
        "Combinatorics"
    ],

    ordered=True

)

dataset["Difficulty"] = pd.Categorical(

    dataset["Difficulty"],

    categories=[
        "Easy",
        "Medium",
        "Hard"
    ],

    ordered=True

)

dataset = dataset.sort_values(
    ["Category", "Difficulty"]
).reset_index(drop=True)

dataset.insert(
    0,
    "Question Number",
    range(1, len(dataset) + 1)
)
# ==========================================================
# FORMAT TEXT
# ==========================================================

dataset["Question"] = dataset["Question"].fillna("").apply(
    lambda x: "\n".join(
        textwrap.wrap(str(x), width=70)
    )
)

dataset["Ground Truth Solution"] = dataset["Ground Truth Solution"].fillna("").apply(
    lambda x: "\n".join(
        textwrap.wrap(str(x), width=90)
    )
)
dataset["Ground Truth Final Answer"] = (
    dataset["Ground Truth Final Answer"]
    .fillna("")
    .astype(str)
)

# ==========================================================
# SAVE
# ==========================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    dataset.to_excel(
        writer,
        index=False,
        sheet_name="Final Dataset"
    )

    worksheet = writer.sheets["Final Dataset"]

    # Freeze header
    worksheet.freeze_panes = "A2"

    # Enable filter
    worksheet.auto_filter.ref = worksheet.dimensions

    # Column widths
    worksheet.column_dimensions["A"].width = 15   # Question Number
    worksheet.column_dimensions["B"].width = 18   # Question ID
    worksheet.column_dimensions["C"].width = 15   # Source
    worksheet.column_dimensions["D"].width = 20   # Category
    worksheet.column_dimensions["E"].width = 12   # Difficulty
    worksheet.column_dimensions["F"].width = 70   # Question
    worksheet.column_dimensions["G"].width = 20   # Ground Truth Final Answer
    worksheet.column_dimensions["H"].width = 90   # Ground Truth Solution

    # Wrap text
    for row in worksheet.iter_rows():

        max_lines = 1

        for cell in row:

            cell.alignment = Alignment(
                wrap_text=True,
                vertical="top"
            )

            if cell.value:

                line_count = str(cell.value).count("\n") + 1

                max_lines = max(
                    max_lines,
                    line_count
                )

        worksheet.row_dimensions[row[0].row].height = max(
            25,
            max_lines * 18
        )

    # Center align selected columns
    for col in ["A", "B", "C", "D", "E", "F", "G"]:

        for cell in worksheet[col]:

            cell.alignment = Alignment(
                horizontal="center",
                vertical="top",
                wrap_text=True
            )

print("=" * 60)
print("Final Dataset Created")
print(OUTPUT_FILE)
print("=" * 60)

print()

print(
    dataset.groupby(
        ["Category", "Difficulty"]
    ).size()
)