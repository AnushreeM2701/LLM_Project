import os
import time
import pandas as pd

from google.genai.errors import ClientError

from app.config import (
    MODELS,
    PROMPTS,
    DATASET_PATH,
    RESULTS_PATH
)

from app.models.inference import generate_response
from app.parser.response_parser import parse_response
from app.evaluation.evaluator import evaluate_response
from app.utils.csv_writer import append_result


# ==========================================================
# Load Completed Experiments
# ==========================================================

def load_completed_experiments(path):
    """
    Read the existing results file so that
    interrupted experiments can resume.
    """

    if not os.path.exists(path):
        return set()

    df = pd.read_csv(path)

    completed = set()

    for _, row in df.iterrows():

        completed.add(

            (
                row["Question Number"],
                row["Model"],
                row["Prompt"]
            )

        )

    return completed


# ==========================================================
# Run Experiments
# ==========================================================

def run_experiment():

    dataset = pd.read_csv(DATASET_PATH)

    completed = load_completed_experiments(
        RESULTS_PATH
    )

    total_experiments = (
        len(dataset)
        * len(MODELS)
        * len(PROMPTS)
    )

    current = len(completed)

    print("=" * 60)
    print("Starting Experiments")
    print("=" * 60)
    print(f"Total Experiments : {total_experiments}")

    for model in MODELS:

        print(f"\nModel : {model}")

        for prompt in PROMPTS:

            print(f"\nPrompt : {prompt}")

            for _, row in dataset.iterrows():

                question_number = row["Question Number"]

                experiment = (
                    question_number,
                    model,
                    prompt
                )

                # --------------------------------------
                # Resume Support
                # --------------------------------------

                if experiment in completed:

                    print(
                        f"Skipping Question {question_number}"
                    )

                    continue

                question = row["Question"]

                original_level = row["Original Level"]

                difficulty = row["Difficulty"]

                ground_truth_solution = row[
                    "Ground Truth Solution"
                ]

                ground_truth_answer = row[
                    "Ground Truth Answer"
                ]

                print(
                    f"\nRunning Question {question_number}"
                )

                # --------------------------------------
                # Measure Execution Time
                # --------------------------------------

                start_clock = time.perf_counter()

                start_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                while True:

                    try:

                        response = generate_response(

                            model_name=model,

                            prompt_type=prompt,

                            question=question

                        )

                        break

                    except ClientError as e:

                        if "429" in str(e):

                            print(
                                "\nQuota exceeded."
                            )

                            print(
                                "Waiting 60 seconds..."
                            )

                            time.sleep(60)

                        else:

                            raise
                # --------------------------------------
                # End Time
                # --------------------------------------

                end_clock = time.perf_counter()

                end_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                execution_time = round(
                    end_clock - start_clock,
                    2
                )

                # --------------------------------------
                # Parse Model Response
                # --------------------------------------

                parsed = parse_response(response)

                # --------------------------------------
                # Evaluate Answer
                # --------------------------------------

                evaluation = evaluate_response(

                    ground_truth_answer,

                    parsed["model_final_answer"]

                )
                execution_time = round(
                    end_clock - start_clock,
                    3
                )
                # --------------------------------------
                # Create Result
                # --------------------------------------

                result = {

                    "Experiment ID":
                        f"Q{question_number}_{model}_{prompt}",

                    "Question Number":
                        question_number,

                    "Question":
                        question,

                    "Original Level":
                        original_level,

                    "Difficulty":
                        difficulty,

                    "Model":
                        model,

                    "Prompt":
                        prompt,

                    "Ground Truth Solution":
                        ground_truth_solution,

                    "Ground Truth Answer":
                        ground_truth_answer,

                    "Model Response":
                        parsed["model_response"],

                    "Model Final Answer":
                        parsed["model_final_answer"],

                    "Answer Correct":
                        evaluation["answer_correct"],

                    "Error Type":
                        evaluation["error_type"],

                    "Start Time":
                        start_time,

                    "End Time":
                        end_time,

                    "Execution Time (s)":
                        execution_time
                }

                # --------------------------------------
                # Save Immediately
                # --------------------------------------

                append_result(

                    result,

                    RESULTS_PATH

                )

                completed.add(experiment)

                current += 1

                print(

                    f"Completed "

                    f"{current}/{total_experiments}"

                )

    print("\n" + "=" * 60)

    print("All Experiments Completed")

    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    run_experiment()
            