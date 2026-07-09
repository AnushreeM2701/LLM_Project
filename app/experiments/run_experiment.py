import os
import time
import pandas as pd
MAX_RETRIES = 5
WAIT_TIME = 60
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
                row["Question ID"],
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
        skip_model = False
        for prompt in PROMPTS:
            if skip_model:
                break
            print(f"\nPrompt : {prompt}")

            for _, row in dataset.iterrows():

                question_number = row["Question Number"]
                question_id = row["Question ID"]
                experiment = (
                    question_id,
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

                retry_count = 0

                while True:

                    try:

                        response = generate_response(

                            model_name=model,

                            prompt_type=prompt,

                            question=question

                        )

                        break

                    except Exception as e:

                        error = str(e)

                        # ------------------------
                        # Daily Quota
                        # ------------------------

                        if "429" in error:

                            retry_count += 1

                            print(
                                f"\nQuota exceeded "
                                f"({retry_count}/{MAX_RETRIES})"
                            )

                            if retry_count >= MAX_RETRIES:

                                print(
                                    f"\nSkipping remaining "
                                    f"{model} experiments."
                                )

                                skip_model = True

                                break

                            print(
                                f"Waiting {WAIT_TIME} seconds..."
                            )

                            time.sleep(WAIT_TIME)

                        # ------------------------
                        # Temporary Server Busy
                        # ------------------------

                        elif "503" in error:

                            retry_count += 1

                            print(
                                f"\nServer Busy "
                                f"({retry_count}/{MAX_RETRIES})"
                            )

                            if retry_count >= MAX_RETRIES:

                                print(
                                    "\nSkipping this question."
                                )

                                response = ""

                                break
                            WAIT_SERVER = 30
                            time.sleep(WAIT_SERVER)

                        else:

                            raise

                if skip_model:
                    break

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
                # -----------------------------------
                # End Time
                # -----------------------------------

                end_clock = time.perf_counter()

                end_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S"
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

                    "Question ID":
                        question_id,

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
            