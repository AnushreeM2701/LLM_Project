import os
import time
import datetime
import re
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
from app.utils.csv_writer import append_result


def _normalize_answer(x):
    """Normalize answers for exact-match evaluation."""
    if x is None:
        return ""
    s = str(x).strip()
    # remove LaTeX wrapper
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s).strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # strip trailing punctuation
    s = s.rstrip(". ,;:)")
    return s



def load_completed_experiments(path):
    """
    Load completed experiments from the CSV.
    This enables resume capability.
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


def run_experiment():

    dataset = pd.read_csv(DATASET_PATH)

    completed = load_completed_experiments(RESULTS_PATH)

    total_experiments = (
        len(dataset)
        * len(MODELS)
        * len(PROMPTS)
    )

    current = len(completed)

    print("=" * 60)
    print(f"Total Experiments : {total_experiments}")
    print("=" * 60)

    for model in MODELS:

        print(f"\nRunning Model : {model}")

        for prompt in PROMPTS:

            print(f"\nPrompt : {prompt}")

            for _, row in dataset.iterrows():

                question_number = row["Question Number"]

                experiment = (
                    question_number,
                    model,
                    prompt
                )

                # Skip completed experiments
                if experiment in completed:

                    print(f"Skipping Question {question_number}")

                    continue

                question = row["Question"]

                level = row["Original Level"]

                difficulty = row["Difficulty"]

                ground_truth_solution = row["Ground Truth Solution"]

                ground_truth_answer = row["Ground Truth Answer"]

                print(f"Running Question {question_number}")

                # Retry if API quota/rate limit is reached
                while True:

                    try:

                        start_time = time.time()
                        start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        response = generate_response(
                            model_name=model,
                            prompt_type=prompt,
                            question=question
                        )

                        end_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        elapsed = time.time() - start_time

                        break

                    except ClientError as e:

                        if "429" in str(e):

                            print("\nQuota exceeded.")
                            print("Waiting 60 seconds...\n")

                            raise

                        else:
                            raise

                parsed = parse_response(response)

                pred = _normalize_answer(parsed.get("final_answer", ""))
                gt = _normalize_answer(ground_truth_answer)

                if not pred:
                    correct_val = "False"
                    error_type_val = "No Final Answer"
                else:
                    if pred == gt:
                        correct_val = "True"
                        error_type_val = ""
                    else:
                        correct_val = "False"
                        error_type_val = "Incorrect Answer"

                result = {


                    "Experiment ID":
                        f"{question_number}_{model}_{prompt}",

                    "Timestamp":
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                    "Start Time":
                        start_dt,

                    "End Time":
                        end_dt,

                    "Time Taken Seconds":
                        round(elapsed, 6),


                    "Question Number":
                        question_number,

                    "Original Level":
                        level,

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

                    "Full Response":
                        parsed["full_response"],

                    "Reasoning":
                        parsed["reasoning"],

                    "Final Answer":
                        parsed["final_answer"],

                    "Step Count":
                        parsed["step_count"],

                    "Correct":
                        correct_val,

                    "Error Type":
                        error_type_val


                }

                append_result(
                    result,
                    RESULTS_PATH
                )

                completed.add(experiment)

                current += 1

                print(f"Progress : {current}/{total_experiments}")

    print("\n" + "=" * 60)
    print("All Experiments Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":

    run_experiment()