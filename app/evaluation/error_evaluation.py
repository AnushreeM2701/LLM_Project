import json
import os
import time

import pandas as pd

from app.models.model_loader import get_model
from app.utils.csv_to_excel import csv_to_excel

# ==========================================================
# FILE PATHS
# ==========================================================

RESULTS_PATH = "data/results/experiment_results.csv"

CHECKPOINT_INTERVAL = 1

MAX_RETRIES = 5

WAIT_TIME = 30


# ==========================================================
# ERROR TAXONOMY
# ==========================================================

ERROR_TYPES = [

    "Correct",

    "Arithmetic Error",

    "Algebraic Manipulation Error",

    "Probability Reasoning Error",

    "Combinatorial Counting Error",

    "Number Theory Error",

    "Formula Misapplication",

    "Incorrect Assumption",

    "Logical Reasoning Error",

    "Incomplete Reasoning",

    "Answer Extraction Error",

    "Other"

]


# ==========================================================
# LOAD RESULTS
# ==========================================================

print()

print("=" * 60)
print("Loading Experiment Results...")
print("=" * 60)

results = pd.read_csv(RESULTS_PATH)


# ==========================================================
# CREATE NEW COLUMNS
# ==========================================================

required_columns = [

    "Error Step",

    "Error Type",

    "Confidence",

    "Justification"

]

for column in required_columns:

    if column not in results.columns:

        results[column] = ""

# Force text columns
results["Error Type"] = results["Error Type"].fillna("").astype(str)
results["Confidence"] = results["Confidence"].fillna("").astype(str)
results["Justification"] = results["Justification"].fillna("").astype(str)

# Error Step can stay numeric/object
results["Error Step"] = results["Error Step"].astype(object)


# ==========================================================
# LOAD JUDGE MODEL
# ==========================================================

JUDGE_MODEL = "gemini"

judge = get_model(JUDGE_MODEL)
# ==========================================================
# BUILD PROMPT
# ==========================================================

def build_error_prompt(

    question,

    category,

    ground_truth,

    model_answer,

    reasoning

):

    prompt = f"""
You are an expert mathematics examiner.

Your task is to evaluate the mathematical reasoning produced by an AI model.

You are acting as an independent mathematics examiner.

Do NOT compare the reasoning with any reference solution or preferred solving method.

Instead, verify whether the mathematical reasoning itself is logically and mathematically valid.

Evaluate the reasoning exactly as written.

Identify the FIRST reasoning step where the mathematical logic becomes invalid.

If the parser failed to extract the model's final answer
or the final answer field is empty,
classify it as "Answer Extraction Error".

If the model explicitly states an incorrect final answer,
DO NOT classify it as "Answer Extraction Error".

Instead,
identify the first mathematical reasoning error
that led to the incorrect final answer.

Question

{question}

Category

{category}
Category-specific guidance

Probability:
Check independence, conditional probability, counting, and probability laws.

Combinatorics:
Check counting principles, permutations, combinations, and overcounting.

Algebra:
Check algebraic manipulation, simplification, and equation solving.

Number Theory:
Check divisibility, modular arithmetic, parity, and integer properties.

Correct Final Answer

{ground_truth}

Model Reasoning

{reasoning}

Model Final Answer

{model_answer}

The parser has already extracted the model's final answer.

Do NOT assume extraction failed unless the
Model Final Answer field is empty.

Return ONLY valid JSON.

{{
    "error_step": integer,

    "error_type": "...",

    "confidence": "...",

    "justification": "..."
}}

Rules

error_step

Count ONLY major mathematical reasoning steps.

Do NOT count explanations, repeated statements, or formatting.

If the reasoning is explicitly numbered, use those numbers.

If it is not numbered, count each major mathematical calculation or logical inference as one step.

Return the first incorrect reasoning step.

If there is no reasoning error and only the final answer extraction failed, return -1.

error_type must be exactly one of

{ERROR_TYPES}

confidence

High

Medium

Low

justification

Write ONLY one concise sentence (maximum 25 words).

Do NOT include multiple reasons.

Do NOT explain the whole solution.

Do NOT include markdown.

Do NOT include extra text before or after the JSON.

Return ONLY the JSON object.
"""

    return prompt

# ==========================================================
# CLASSIFY ONE ERROR
# ==========================================================

def classify_error(

    question,

    category,

    ground_truth,

    model_answer,

    reasoning

):

    prompt = build_error_prompt(

        question,

        category,

        ground_truth,

        model_answer,

        reasoning

    )

    retries = 0

    while retries < MAX_RETRIES:

        try:

            response = judge(prompt)

            response = response.strip()

            # ---------------------------------------
            # Remove Markdown if Gemini returns it
            # ---------------------------------------

            response = response.replace(
                "```json",
                ""
            )

            response = response.replace(
                "```",
                ""
            )

            response = response.strip()

            start = response.find("{")

            end = response.rfind("}")

            if start != -1 and end != -1:

                response = response[start:end+1]

            result = json.loads(response)

            return {

                "Error Step":
                    result.get(
                        "error_step",
                        ""
                    ),

                "Error Type":
                    result.get(
                        "error_type",
                        "Other"
                    ),

                "Confidence":
                    result.get(
                        "confidence",
                        "Low"
                    ),

                "Justification":
                    result.get(
                        "justification",
                        ""
                    )

            }

        except Exception as e:

            retries += 1

            print()

            print(
                f"Classification failed "
                f"({retries}/{MAX_RETRIES})"
            )

            print(str(e))

            time.sleep(WAIT_TIME)

    # ---------------------------------------
    # Failed after retries
    # ---------------------------------------

    return {

        "Error Step": "",

        "Error Type": "Other",

        "Confidence": "Low",

        "Justification":
            "Classification failed."

    }


# ==========================================================
# SAVE PROGRESS
# ==========================================================

def save_progress(df):

    df.to_csv(

        RESULTS_PATH,

        index=False

    )

    csv_to_excel(

        RESULTS_PATH

    )

    print()

    print("=" * 60)

    print("Progress Saved")

    print("=" * 60)

# ==========================================================
# RUN ERROR EVALUATION
# ==========================================================

def run_error_evaluation():

    print()
    print("=" * 60)
    print("Starting Error Evaluation")
    print("=" * 60)

    total_errors = len(
        results[
            results["Answer Correct"] == False
        ]
    )

    completed = 0

    for index, row in results.iterrows():

        # ---------------------------------------------
        # Skip Correct Answers
        # ---------------------------------------------

        if row["Answer Correct"] == True:

            if row["Error Type"] == "":

                results.at[
                    index,
                    "Error Type"
                ] = "Correct"

            continue

        # ---------------------------------------------
        # Skip Already Classified
        # ---------------------------------------------

        if pd.notna(row["Confidence"]):

            if str(row["Confidence"]).strip() != "":

                completed += 1

                continue

        print()

        print(
            f"Question {row['Question Number']} | "
            f"{row['Model']} | "
            f"{row['Prompt']}"
        )
        if str(row["Model Final Answer"]).strip() == "":

            results.at[index, "Error Type"] = "Answer Extraction Error"

            results.at[index, "Confidence"] = "High"

            results.at[index, "Justification"] = (
                "Parser failed to extract the final answer."
            )

            completed += 1

            continue
        classification = classify_error(

            question=row["Question"],

            category=row["Category"],

            ground_truth=row["Ground Truth Final Answer"],

            model_answer=row["Model Final Answer"],

            reasoning=row["Model Response"]

        )
        
        results.at[
            index,
            "Error Step"
        ] = classification["Error Step"]

        results.at[
            index,
            "Error Type"
        ] = classification["Error Type"]

        results.at[
            index,
            "Confidence"
        ] = classification["Confidence"]

        results.at[
            index,
            "Justification"
        ] = classification["Justification"]

        completed += 1

        print(
            f"Completed "
            f"{completed}/{total_errors}"
        )

        # ---------------------------------------------
        # Checkpoint Save
        # ---------------------------------------------

        if completed % CHECKPOINT_INTERVAL == 0:

            save_progress(results)

    # ---------------------------------------------
    # Final Save
    # ---------------------------------------------

    save_progress(results)

    print()

    print("=" * 60)
    print("Error Evaluation Completed")
    print("=" * 60)

    print()

    print(
        results[
            "Error Type"
        ].value_counts()
    )
print(results["Error Type"].value_counts())

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    run_error_evaluation()