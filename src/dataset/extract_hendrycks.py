"""
Pulls Easy/Medium candidate questions from the Hendrycks MATH dataset
(EleutherAI/hendrycks_math on Hugging Face) into
config.HENDRYCKS_CANDIDATES_PATH.

Ported from the prior pipeline's app/dataset/create_easy_medium_dataset.py,
logic unchanged. Not re-run as part of this rebuild —
data/curated/hendrycks_candidates.xlsx already contains the prior run's
output plus the manual Include/Final Category curation done on top of it.

NOTE (see docs/methodology.md and docs/limitations.md): Hendrycks MATH is a
widely-used public benchmark and is very likely present in the pretraining
data of all three study models — a genuine contamination risk for Easy/
Medium accuracy. Hard-tier questions are sourced from AIME instead
specifically to reduce this risk (see freeze_dataset.py).
"""

import os
import re

import pandas as pd
from datasets import load_dataset

from config.config import HENDRYCKS_CANDIDATES_PATH

SUBJECTS = {
    "Algebra": "algebra",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Counting & Probability": "counting_and_probability",
}

# Level 5 ("Hard") is intentionally excluded here — the Hard tier is
# AIME-only in this pipeline (see freeze_dataset.py for the rationale).
LEVEL_MAP = {
    "Level 1": "Easy",
    "Level 2": "Easy",
    "Level 3": "Medium",
    "Level 4": "Medium",
}


def clean_text(text):

    if text is None:
        return ""

    text = str(text)
    text = text.replace("$", "")

    for item in [r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\quad", r"\qquad", r"\displaystyle", r"\textstyle"]:
        text = text.replace(item, "")

    replace = {
        r"\times": "\u00d7", r"\cdot": "\u00b7", r"\le": "\u2264", r"\ge": "\u2265",
        r"\neq": "\u2260", r"\ne": "\u2260", r"\pi": "\u03c0", r"\theta": "\u03b8",
        r"\alpha": "\u03b1", r"\beta": "\u03b2", r"\gamma": "\u03b3", r"\omega": "\u03c9",
        r"\infty": "\u221e", r"\sqrt": "\u221a",
    }
    for k, v in replace.items():
        text = text.replace(k, v)

    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1/\2)", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_final_answer(solution):

    solution = str(solution)
    start = solution.rfind(r"\boxed{")
    if start == -1:
        return ""

    i = start + len(r"\boxed{")
    brace_count = 1
    answer = ""

    while i < len(solution):
        char = solution[i]
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                break
        answer += char
        i += 1

    return answer.strip()


def extract_hendrycks_candidates():

    sheet_data = {"Algebra": [], "Number Theory": [], "Counting & Probability": []}

    for subject, config_name in SUBJECTS.items():

        print(f"Loading {subject}...")

        dataset = load_dataset("EleutherAI/hendrycks_math", config_name)
        train = dataset["train"]

        final_category = "Algebra" if subject == "Intermediate Algebra" else subject

        for item in train:

            if item["level"] not in LEVEL_MAP:
                continue

            difficulty = LEVEL_MAP[item["level"]]

            sheet_data[final_category].append({
                "Category": final_category,
                "Difficulty": difficulty,
                "Original Subject": subject,
                "Original Level": item["level"],
                "Question": clean_text(item["problem"]),
                "Ground Truth Final Answer": extract_final_answer(item["solution"]),
                "Ground Truth Solution": clean_text(item["solution"]),
                "Include": "F",
                "Final Category": "",
            })

    os.makedirs(os.path.dirname(HENDRYCKS_CANDIDATES_PATH), exist_ok=True)

    with pd.ExcelWriter(HENDRYCKS_CANDIDATES_PATH, engine="openpyxl") as writer:
        for sheet, rows in sheet_data.items():
            df = pd.DataFrame(rows)
            df["Difficulty"] = pd.Categorical(df["Difficulty"], categories=["Easy", "Medium"], ordered=True)
            df = df.sort_values(["Difficulty", "Question"])
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Wrote Hendrycks candidates -> {HENDRYCKS_CANDIDATES_PATH}")
    print("NOTE: newly extracted rows need manual Include/Final Category "
          "curation before they can enter the dataset.")


if __name__ == "__main__":
    extract_hendrycks_candidates()
