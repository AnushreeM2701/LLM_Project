"""
RQ3: Does ToT reduce final-answer errors compared to CoT?

Per-model McNemar's exact test on paired (same question) CoT vs ToT
correctness — not just raw accuracy percentages. During the methodology
review this distinction mattered: raw percentages alone suggested "all
three models show a small, similar prompt effect," but the paired test
showed only one model's CoT-over-ToT gap was actually distinguishable from
chance (see docs/limitations.md for the full result once this is re-run on
new-model data).
"""

import os

import pandas as pd

from config.config import TABLES_DIR, MODEL_NAMES
from src.utils.io import load_results
from src.utils.stats import mcnemar_exact, paired_accuracy_discordant_counts, wilson_ci


def run() -> pd.DataFrame:

    df = load_results()
    rows = []

    for model in MODEL_NAMES:

        model_df = df[df["Model"] == model]

        cot_df = model_df[model_df["Prompt"] == "cot"]
        tot_df = model_df[model_df["Prompt"] == "tot"]

        if len(cot_df) == 0 or len(tot_df) == 0:
            continue

        cot_acc = wilson_ci(int(cot_df["Answer Correct"].astype(bool).sum()), len(cot_df))
        tot_acc = wilson_ci(int(tot_df["Answer Correct"].astype(bool).sum()), len(tot_df))

        b, c = paired_accuracy_discordant_counts(
            model_df, condition_col="Prompt", condition_a="cot", condition_b="tot",
            pair_key_col="Question ID",
        )

        test = mcnemar_exact(b, c)

        rows.append({
            "Model": model,
            "CoT Accuracy": cot_acc["proportion"],
            "CoT N": cot_acc["n"],
            "ToT Accuracy": tot_acc["proportion"],
            "ToT N": tot_acc["n"],
            "CoT-only Correct (b)": b,
            "ToT-only Correct (c)": c,
            "McNemar p-value": test["p_value"],
            "Significant (p<0.05)": test["significant"],
            "Direction": "CoT > ToT" if b > c else ("ToT > CoT" if c > b else "No difference"),
        })

    result = pd.DataFrame(rows)

    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, "rq3_prompt_comparison.csv")
    result.to_csv(path, index=False)

    print(f"Saved -> {path}")
    print(result)

    return result


if __name__ == "__main__":
    run()
