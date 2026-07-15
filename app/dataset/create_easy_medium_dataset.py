from datasets import load_dataset
import pandas as pd
import os
import re

# ==========================================================
# CONFIG
# ==========================================================

SUBJECTS = {
    "Algebra": "algebra",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Counting & Probability": "counting_and_probability"
}

LEVEL_MAP = {
    "Level 1": "Easy",
    "Level 2": "Easy",
    "Level 3": "Medium"
}

OUTPUT_FOLDER = "data/candidates"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "easy_medium_candidates.xlsx"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LATEX CLEANER
# ==========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace("$", "")

    remove = [
        r"\left",
        r"\right",
        r"\!",
        r"\,",
        r"\;",
        r"\:",
        r"\quad",
        r"\qquad",
        r"\displaystyle",
        r"\textstyle"
    ]

    for item in remove:
        text = text.replace(item, "")

    replace = {
        r"\times":"×",
        r"\cdot":"·",
        r"\le":"≤",
        r"\ge":"≥",
        r"\neq":"≠",
        r"\ne":"≠",
        r"\pi":"π",
        r"\theta":"θ",
        r"\alpha":"α",
        r"\beta":"β",
        r"\gamma":"γ",
        r"\omega":"ω",
        r"\infty":"∞",
        r"\sqrt":"√"
    }

    for k,v in replace.items():
        text = text.replace(k,v)

    text = re.sub(
        r"\\boxed\{([^}]*)\}",
        r"\1",
        text
    )

    text = re.sub(
        r"\\frac\{([^}]*)\}\{([^}]*)\}",
        r"(\1/\2)",
        text
    )

    text = text.replace("{","")
    text = text.replace("}","")

    text = re.sub(r"\s+"," ",text)

    return text.strip()
# ==========================================================
# Extract Ground Truth Answer
# ==========================================================

def extract_final_answer(solution):

    solution = str(solution)

    start = solution.rfind(r"\boxed{")

    if start == -1:
        return ""

    # Move to first character after \boxed{
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
# ==========================================================
# STORAGE
# ==========================================================

sheet_data = {
    "Algebra":[],
    "Number Theory":[],
    "Counting & Probability":[]
}

# ==========================================================
# LOAD DATASETS
# ==========================================================

for subject, config in SUBJECTS.items():

    print(f"Loading {subject}...")

    dataset = load_dataset(
        "EleutherAI/hendrycks_math",
        config
    )

    train = dataset["train"]

    final_category = (
        "Algebra"
        if subject=="Intermediate Algebra"
        else subject
    )

    for item in train:

        if item["level"] not in LEVEL_MAP:
            continue

        difficulty = LEVEL_MAP[item["level"]]

        sheet_data[final_category].append({

            "Category":final_category,

            "Difficulty":difficulty,

            "Original Subject":subject,

            "Original Level":item["level"],

            "Question":clean_text(item["problem"]),

            "Ground Truth Final Answer": extract_final_answer(item["solution"]),

            "Ground Truth Solution":clean_text(item["solution"]),

            "Include":"F",

            "Final Category":""

        })

# ==========================================================
# SAVE EXCEL
# ==========================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    for sheet, rows in sheet_data.items():

        df = pd.DataFrame(rows)

        df["Difficulty"] = pd.Categorical(
            df["Difficulty"],
            categories=["Easy","Medium"],
            ordered=True
        )

        df = df.sort_values(
            ["Difficulty", "Question"]
        )

        df.to_excel(
            writer,
            sheet_name=sheet,
            index=False
        )

print("\n" + "="*60)
print("Workbook Created Successfully")
print(OUTPUT_FILE)
print("="*60)