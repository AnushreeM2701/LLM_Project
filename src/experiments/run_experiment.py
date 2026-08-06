"""
Resumable, checkpointed experiment runner.

Ports the prior pipeline's retry/backoff/resume pattern (found sound during
review) essentially unchanged, adapted for: (a) three new models, (b) real
multi-branch ToT instead of a single call, (c) proactive rate-limit pacing
per provider (Mistral's free tier is 2 req/min), (d) provenance columns
(served model version, prompt version) on every row via src/utils/io.py,
and (e) running all three models CONCURRENTLY (one thread each).

(e) matters a lot in practice: Mistral's 2 req/min cap makes it the slowest
model by a wide margin (~5-6 hours for the full dataset vs. Gemini/Groq's
tens of minutes). Running models sequentially means Gemini and Groq's time
is pure dead weight added on top of Mistral's; running them concurrently
means total wall-clock time is bounded by the SLOWEST model, not the SUM of
all three. Each model has its own independent rate limit (config.
RETRY_SETTINGS) and writes through the same thread-safe append_result
(src/utils/io.py), so this is a safe, meaningful speedup, not a shortcut
that risks correctness.

Run: python -m src.experiments.run_experiment
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config.config import MODEL_NAMES, PROMPT_TYPES, PROMPT_VERSIONS
from src.utils.io import load_dataset, load_completed_experiments, append_result
from src.experiments.generation import generate_with_retry, QuotaExhausted
from src.parser.response_parser import parse_response
from src.evaluation.answer_evaluator import evaluate_response


class ProgressCounter:
    """Thread-safe counter shared across the 3 concurrent model threads,
    so "every 10 completions" is counted across ALL models combined, not
    per-model (which would fire at wildly different real times given
    Mistral's 2 req/min vs. Gemini/Groq's much faster pacing). Prints a
    flushed CHECKPOINT line the caller can watch for via the Monitor tool
    -- flush=True matters here since Python fully buffers stdout when it
    isn't a terminal (i.e. whenever this runs under Monitor/background)."""

    def __init__(self, total: int, report_every: int = 10):
        self.count = 0
        self.total = total
        self.report_every = report_every
        self.lock = threading.Lock()
        self.start_time = time.perf_counter()

    def increment(self) -> None:
        with self.lock:
            self.count += 1
            n = self.count
            if n % self.report_every == 0 or n == self.total:
                elapsed_min = (time.perf_counter() - self.start_time) / 60
                rate_per_min = n / elapsed_min if elapsed_min > 0 else 0
                remaining = self.total - n
                eta_min = remaining / rate_per_min if rate_per_min > 0 else float("inf")
                print(
                    f"CHECKPOINT: {n}/{self.total} completed "
                    f"({elapsed_min:.1f} min elapsed, "
                    f"~{eta_min:.1f} min remaining at current rate)",
                    flush=True,
                )


def run_model(model: str, dataset: pd.DataFrame, completed: set, progress: ProgressCounter) -> int:
    """Runs every (prompt_type, question) combination for ONE model.
    Returns the count of newly-completed experiments. `completed` is only
    ever mutated with keys for THIS model by THIS thread -- other models'
    keys are disjoint, so no cross-thread write conflicts occur despite
    `completed` being a shared set object."""

    print(f"\n[{model}] starting")
    skip_model = False
    newly_completed = 0

    for prompt_type in PROMPT_TYPES:

        if skip_model:
            break

        for _, row in dataset.iterrows():

            question_id = row["Question ID"]
            experiment_key = (question_id, model, prompt_type)

            if experiment_key in completed:
                continue

            question = row["Question"]

            start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            start_clock = time.perf_counter()

            try:
                generation = generate_with_retry(model, prompt_type, question)
            except QuotaExhausted:
                skip_model = True
                break

            end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            execution_time = round(time.perf_counter() - start_clock, 3)

            parsed = parse_response(generation["final_text"])
            evaluation = evaluate_response(
                row["Ground Truth Final Answer"], parsed["model_final_answer"]
            )

            result = {
                "Experiment ID": f"{question_id}_{model}_{prompt_type}",
                "Question Number": row["Question Number"],
                "Question ID": question_id,
                "Category": row["Category"],
                "Question": question,
                "Difficulty": row["Difficulty"],
                "Model": model,
                "Model Version": generation["model_version"],
                "Prompt": prompt_type,
                "Prompt Version": PROMPT_VERSIONS[prompt_type],
                "Ground Truth Solution": row["Ground Truth Solution"],
                "Ground Truth Answer": row["Ground Truth Final Answer"],
                "Model Response": parsed["model_response"],
                "Model Final Answer": parsed["model_final_answer"],
                "Reasoning": parsed["reasoning"],
                "Step Count": parsed["model_step_count"],
                "Answer Correct": evaluation["correct"],
                "Start Time": start_time,
                "End Time": end_time,
                "Execution Time (s)": execution_time,
                "Error Step": "",
                "Error Type": "",
                "Error Subtype": "",
                "Confidence": "",
                "Justification": "",
                "Judge Model": "",
            }

            append_result(result)
            completed.add(experiment_key)
            newly_completed += 1

            print(f"[{model}] {question_id}/{prompt_type}: "
                  f"{'correct' if evaluation['correct'] else 'wrong'} "
                  f"({newly_completed} done this run)")

            progress.increment()

    print(f"\n[{model}] finished this run "
          f"({'quota-limited' if skip_model else 'all done'})")

    return newly_completed


def run_experiment(difficulty_filter: list = None, report_every: int = 10):
    """difficulty_filter: e.g. ["Easy"] to run only the Easy tier as one
    batch of a staged rollout. None runs the whole dataset. Resumable
    either way -- questions already in the canonical results file are
    skipped regardless of which batch they were originally collected in."""

    dataset = load_dataset()

    if difficulty_filter:
        dataset = dataset[dataset["Difficulty"].isin(difficulty_filter)]

    completed = load_completed_experiments()

    all_combos = [
        (row["Question ID"], model, prompt_type)
        for _, row in dataset.iterrows()
        for model in MODEL_NAMES
        for prompt_type in PROMPT_TYPES
    ]
    remaining_this_batch = sum(1 for c in all_combos if c not in completed)

    print("=" * 60)
    print("Starting Experiments")
    print("=" * 60)
    print(f"Batch : {difficulty_filter or 'ALL difficulties'}")
    print(f"Questions in batch : {len(dataset)}")
    print(f"Remaining to run this batch : {remaining_this_batch} "
          f"(of {len(all_combos)} total combos in this batch)")
    print(f"Running {len(MODEL_NAMES)} models concurrently (one thread each)...")

    progress = ProgressCounter(total=remaining_this_batch, report_every=report_every)

    with ThreadPoolExecutor(max_workers=len(MODEL_NAMES)) as executor:

        futures = {
            executor.submit(run_model, model, dataset, completed, progress): model
            for model in MODEL_NAMES
        }

        failed_models = []

        for future in as_completed(futures):
            model = futures[future]
            try:
                future.result()
            except Exception as e:
                # Don't let one model's unhandled error discard the other
                # models' already-collected, already-saved rows. The
                # resumable design means this model's remaining questions
                # just get picked up on the next run.
                print(f"\n[{model}] thread failed (data collected so far is "
                      f"still saved): {e}")
                failed_models.append(model)

        if failed_models:
            print(f"\nModels that hit an unhandled error this run: {failed_models}. "
                  f"Re-run to resume them; already-collected rows are safe.")

    print("\n" + "=" * 60)
    print("All Experiments Completed (or quota-limited for this run)")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
