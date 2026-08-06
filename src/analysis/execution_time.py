"""
Execution time by model/prompt/difficulty. Ported from the prior pipeline's
app/analysis/execution_time*.py, consolidated into one module and fixed to
read the canonical results file exclusively via src.utils.io.load_results()
(the prior pipeline had two of these scripts silently reading a stale
duplicate file — see docs/limitations.md).

Note (see config.config.TOT_BRANCH_COUNT): ToT execution time now includes
all branch-generation calls plus the selection call, so it is expected to
run several times longer than CoT by construction, not as a finding — this
is reported for completeness/transparency, not as an RQ.

Per-question figures (with the Easy/Medium/Hard breakdown) live in
src/analysis/question_execution_time.py -- this module only produces the
summary table.
"""

import os

import pandas as pd

from config.config import TABLES_DIR, MODEL_NAMES, PROMPT_TYPES
from src.utils.io import load_results


def summary_table() -> pd.DataFrame:

    df = load_results()
    rows = []

    for model in MODEL_NAMES:
        for prompt in PROMPT_TYPES:
            for difficulty in ["Easy", "Medium", "Hard"]:

                subset = df[
                    (df["Model"] == model)
                    & (df["Prompt"] == prompt)
                    & (df["Difficulty"] == difficulty)
                ]

                if len(subset) == 0:
                    continue

                times = pd.to_numeric(subset["Execution Time (s)"], errors="coerce").dropna()

                if len(times) == 0:
                    continue

                rows.append({
                    "Model": model,
                    "Prompt": prompt,
                    "Difficulty": difficulty,
                    "N": len(times),
                    "Mean (s)": times.mean(),
                    "Median (s)": times.median(),
                    "Std (s)": times.std(),
                })

    return pd.DataFrame(rows)


def run():

    os.makedirs(TABLES_DIR, exist_ok=True)

    table = summary_table()
    path = os.path.join(TABLES_DIR, "execution_time_summary.csv")
    table.to_csv(path, index=False)
    print(f"Saved -> {path}")
    print(table)

    return table


if __name__ == "__main__":
    run()
