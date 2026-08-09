import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config.config import (
    MODEL_NAMES,
    PROMPT_TYPES,
    DATASET_SEED,
    RESULTS_DIR,
    MODEL_PILOT_DOC_PATH,
)
from src.utils.io import load_dataset
from src.experiments.generation import generate_with_retry, QuotaExhausted
from src.parser.response_parser import parse_response
from src.evaluation.answer_evaluator import evaluate_response

PILOT_RESULTS_PATH = os.path.join(RESULTS_DIR, "pilot_results.csv")

PILOT_COLUMNS = [
    "Question ID", "Difficulty", "Category", "Model", "Model Version", "Prompt",
    "Ground Truth Answer", "Model Final Answer", "Answer Correct",
    "Response Length (chars)", "Leakage Hits", "Latency (s)", "Full Response",
]

PILOT_SAMPLE_PER_DIFFICULTY = {"Hard": 6, "Medium": 3, "Easy": 3}

LEAKAGE_PATTERNS = [
    r"<think>", r"</think>", r"<reasoning>", r"</reasoning>",
    r"<scratchpad>", r"\[thinking\]", r"^okay,? let me think",
]


def select_pilot_sample() -> pd.DataFrame:

    dataset = load_dataset()

    samples = []
    for difficulty, n in PILOT_SAMPLE_PER_DIFFICULTY.items():
        subset = dataset[dataset["Difficulty"] == difficulty]
        n = min(n, len(subset))
        samples.append(subset.sample(n=n, random_state=DATASET_SEED))

    return pd.concat(samples, ignore_index=True)


def detect_leakage(text: str) -> list:
    hits = []
    for pattern in LEAKAGE_PATTERNS:
        if re.search(pattern, text, flags=re.I | re.M):
            hits.append(pattern)
    return hits


def load_completed() -> set:
    if not os.path.exists(PILOT_RESULTS_PATH):
        return set()
    df = pd.read_csv(PILOT_RESULTS_PATH)
    return set(zip(df["Question ID"], df["Model"], df["Prompt"]))


_append_lock = threading.Lock()


def append_pilot_row(row: dict) -> None:
    """Thread-safe -- serializes writes when models run in parallel threads
    (see run_pilot's ThreadPoolExecutor). Same pattern as the lock added to
    src/utils/io.py's append_result after a two-PROCESS race corrupted an
    earlier pilot run; this guards the equivalent two-THREAD case."""

    row_df = pd.DataFrame([row])[PILOT_COLUMNS]

    with _append_lock:
        if os.path.exists(PILOT_RESULTS_PATH):
            row_df.to_csv(PILOT_RESULTS_PATH, mode="a", header=False, index=False)
        else:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            row_df.to_csv(PILOT_RESULTS_PATH, mode="w", header=True, index=False)


def run_model(model: str, sample: pd.DataFrame, completed: set) -> None:
    """Runs every (prompt_type, question) combination for ONE model. Each
    model has its own independent rate limit (see config.RETRY_SETTINGS),
    so running one of these per model concurrently means total wall-clock
    time is bounded by the SLOWEST model (Mistral's 2 req/min), not the SUM
    of all three run sequentially -- previously Gemini and Groq's ~6
    combined minutes were pure dead time added on top of Mistral's ~25."""

    print(f"\n{'=' * 60}\nModel: {model}\n{'=' * 60}")
    quota_exhausted = False

    for prompt_type in PROMPT_TYPES:

        if quota_exhausted:
            break

        for _, row in sample.iterrows():

            question_id = row["Question ID"]

            if (question_id, model, prompt_type) in completed:
                print(f"  [{model}] {question_id}/{prompt_type}: already done, skipping")
                continue

            question = row["Question"]

            try:
                generation = generate_with_retry(model, prompt_type, question)
            except QuotaExhausted:
                quota_exhausted = True
                break

            final_text = generation["final_text"]
            branches = generation["branches"]

            parsed = parse_response(final_text)
            evaluation = evaluate_response(row["Ground Truth Final Answer"], parsed["model_final_answer"])

            leakage_hits = detect_leakage(final_text)
            for branch_text in branches:
                leakage_hits += detect_leakage(branch_text)

            pilot_row = {
                "Question ID": question_id,
                "Difficulty": row["Difficulty"],
                "Category": row["Category"],
                "Model": model,
                "Model Version": generation["model_version"],
                "Prompt": prompt_type,
                "Ground Truth Answer": row["Ground Truth Final Answer"],
                "Model Final Answer": parsed["model_final_answer"],
                "Answer Correct": evaluation["correct"],
                "Response Length (chars)": len(final_text),
                "Leakage Hits": ";".join(sorted(set(leakage_hits))),
                "Latency (s)": generation["latency_s"],
                "Full Response": final_text,
            }

            append_pilot_row(pilot_row)

            status = "correct" if evaluation["correct"] else "wrong"
            leak_flag = " [LEAKAGE?]" if leakage_hits else ""
            print(f"  [{model}] {question_id}/{prompt_type}: {status}{leak_flag}")


def run_pilot():

    sample = select_pilot_sample()
    completed = load_completed()

    print(f"Pilot sample: {len(sample)} questions "
          f"({dict(sample['Difficulty'].value_counts())})")
    print(f"Already completed (resuming): {len(completed)}")
    print(f"Running {len(MODEL_NAMES)} models concurrently (one thread each)...")

    with ThreadPoolExecutor(max_workers=len(MODEL_NAMES)) as executor:

        futures = {
            executor.submit(run_model, model, sample, completed): model
            for model in MODEL_NAMES
        }

        failed_models = []

        for future in as_completed(futures):
            model = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"\n[{model}] thread failed (data collected so far is "
                      f"still saved): {e}")
                failed_models.append(model)

        if failed_models:
            print(f"\nModels that hit an unhandled error this run: {failed_models}. "
                  f"Re-run to resume them; already-collected rows are safe.")

    print(f"\nPilot results -> {PILOT_RESULTS_PATH}")

    if not os.path.exists(PILOT_RESULTS_PATH):
        print("No results collected (all models quota-exhausted immediately).")
        return None

    results_df = pd.read_csv(PILOT_RESULTS_PATH)
    write_summary(results_df)

    return results_df


def write_summary(results_df: pd.DataFrame) -> None:

    lines = ["# Model Pilot Results", ""]
    lines.append(
        "Pilot run before committing to the full-dataset rerun with the new "
        "model roster (gemini-3.5-flash, GPT-OSS-120B via Groq, "
        "mistral-large-latest). See config/config.py for exact settings "
        "(temperature, thinking_level/reasoning_effort) and docs/limitations.md "
        "for why these controls don't fully eliminate the reasoning-model "
        "confound, only hold it constant."
    )
    lines.append("")
    lines.append(
        "Note: this pilot ran on the CORRECTED AIME question text (see "
        "src/dataset/fix_aime_extraction.py) -- the prior pipeline's ~11% "
        "AIME-Hard accuracy figure was measured on corrupted text and old "
        "models both, so it is not a clean baseline for comparison, only a "
        "rough prior indicator."
    )
    lines.append("")
    lines.append("## Accuracy by model/prompt")
    lines.append("")

    acc = (
        results_df.groupby(["Model", "Prompt"])["Answer Correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    lines.append(acc.to_markdown(index=False))
    lines.append("")

    lines.append("## Accuracy on Hard (AIME) only")
    lines.append("")
    hard = results_df[results_df["Difficulty"] == "Hard"]
    if len(hard) > 0:
        hard_acc = hard.groupby(["Model", "Prompt"])["Answer Correct"].agg(["mean", "count"]).reset_index()
        lines.append(hard_acc.to_markdown(index=False))
    else:
        lines.append("No Hard-tier rows collected yet.")
    lines.append("")

    lines.append("## Leaked reasoning-token check")
    lines.append("")
    leaked = results_df[results_df["Leakage Hits"].fillna("") != ""]
    if len(leaked) == 0:
        lines.append(
            "No heuristic leakage indicators (`<think>` tags etc.) found in "
            f"any of the {len(results_df)} pilot responses collected so far."
        )
    else:
        lines.append(f"**{len(leaked)} response(s) flagged** for manual review:")
        lines.append("")
        lines.append(leaked[["Question ID", "Model", "Prompt", "Leakage Hits"]].to_markdown(index=False))
    lines.append("")

    lines.append("## Response length by model (chars)")
    lines.append("")
    length_stats = results_df.groupby("Model")["Response Length (chars)"].describe()[["mean", "min", "max"]]
    lines.append(length_stats.to_markdown())
    lines.append("")

    with open(MODEL_PILOT_DOC_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Summary written -> {MODEL_PILOT_DOC_PATH}")


if __name__ == "__main__":
    run_pilot()
