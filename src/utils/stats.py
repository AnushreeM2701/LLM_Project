from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, fisher_exact, kruskal


# ==========================================================
# RQ3: McNemar's exact test (paired CoT vs ToT accuracy per model)
# ==========================================================

def mcnemar_exact(b: int, c: int) -> dict:
    n = b + c

    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "significant": False}

    result = binomtest(min(b, c), n, 0.5, alternative="two-sided")

    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "p_value": result.pvalue,
        "significant": result.pvalue < 0.05,
    }


def paired_accuracy_discordant_counts(
    df: pd.DataFrame,
    condition_col: str,
    condition_a: str,
    condition_b: str,
    pair_key_col: str,
    correct_col: str = "Answer Correct",
) -> Tuple[int, int]:
    pivot = df.pivot_table(
        index=pair_key_col, columns=condition_col, values=correct_col, aggfunc="first"
    )

    a_correct = pivot[condition_a].astype(bool)
    b_correct = pivot[condition_b].astype(bool)

    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())

    return b, c


# chi-square / Fisher's exact test of independence


def independence_test(contingency_table: pd.DataFrame) -> dict:

    table = contingency_table.to_numpy()

    if table.shape == (2, 2):
        _, expected = chi2_contingency(table)[:2], None
        chi2, p, dof, expected = chi2_contingency(table)
        if (expected < 5).any():
            _, p_fisher = fisher_exact(table)
            return {"test": "fisher_exact", "statistic": None, "p_value": p_fisher, "dof": None}
        return {"test": "chi_square", "statistic": chi2, "p_value": p, "dof": dof}

    chi2, p, dof, expected = chi2_contingency(table)

    min_expected = expected.min()

    return {
        "test": "chi_square",
        "statistic": chi2,
        "p_value": p,
        "dof": dof,
        "min_expected_count": min_expected,
        "caution": min_expected < 5,  # flag, don't silently hide, the approximation getting shaky
    }


# distribution comparison (normalized error step location)

def kruskal_wallis(*groups: Sequence[float]) -> dict:
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return {"statistic": None, "p_value": None, "note": "fewer than 2 non-empty groups"}

    statistic, p_value = kruskal(*groups)

    return {"statistic": statistic, "p_value": p_value, "significant": p_value < 0.05}

# CONFIDENCE INTERVALS

def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> dict:

    if n == 0:
        return {"proportion": None, "lower": None, "upper": None}

    from scipy.stats import norm

    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n

    denominator = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denominator
    margin = (z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)) / denominator

    return {
        "proportion": p_hat,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "n": n,
    }


def bootstrap_ci(
    data: Sequence[float],
    statistic_fn=np.mean,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 25116096,
) -> dict:

    data = np.asarray(data)

    if len(data) == 0:
        return {"estimate": None, "lower": None, "upper": None}

    rng = np.random.default_rng(seed)

    boot_stats = [
        statistic_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]

    alpha = 1 - confidence

    return {
        "estimate": statistic_fn(data),
        "lower": float(np.percentile(boot_stats, 100 * alpha / 2)),
        "upper": float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        "n": len(data),
    }
