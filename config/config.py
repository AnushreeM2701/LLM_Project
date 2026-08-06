"""
Single source of truth for the rebuilt pipeline.

Every path, model setting, and pipeline parameter lives here.
No other module should hardcode a path or model ID.
"""

import os

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CURATED_DIR = os.path.join(DATA_DIR, "curated")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

DOCS_DIR = os.path.join(BASE_DIR, "docs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")

# Canonical single-source-of-truth files. No script may write a copy,
# a "repaired" variant, or a backup with a different name in this directory.
AIME_MASTER_PATH = os.path.join(RAW_DIR, "aime_master.xlsx")
HENDRYCKS_CANDIDATES_PATH = os.path.join(CURATED_DIR, "hendrycks_candidates.xlsx")
FINAL_DATASET_PATH = os.path.join(PROCESSED_DIR, "final_dataset.xlsx")

RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "experiment_results.csv")
RESULTS_XLSX_PATH = os.path.join(RESULTS_DIR, "experiment_results.xlsx")

METHODOLOGY_DOC_PATH = os.path.join(DOCS_DIR, "methodology.md")
LIMITATIONS_DOC_PATH = os.path.join(DOCS_DIR, "limitations.md")
MODEL_PILOT_DOC_PATH = os.path.join(DOCS_DIR, "model_pilot_results.md")

for _dir in (
    RAW_DIR, CURATED_DIR, PROCESSED_DIR, RESULTS_DIR,
    DOCS_DIR, FIGURES_DIR, TABLES_DIR, STATS_DIR,
):
    os.makedirs(_dir, exist_ok=True)

# ==========================================================
# DATASET COMPOSITION
# ==========================================================

CATEGORIES = ["Algebra", "Number Theory", "Probability", "Combinatorics"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]

# Hard tier is AIME-only (Hendrycks Level-5 dropped — see docs/methodology.md
# for the empirical justification: AIME-Hard and Hendrycks-Hard accuracy
# differed by ~80 points under the old pipeline, i.e. they were not measuring
# the same construct). Uses all currently curated+categorized AIME questions,
# unbalanced across category (Probability tops out at 8 available questions;
# trimming the other categories to match would discard curated data for no
# benefit). Category imbalance is handled as a covariate in analysis, not by
# forcing equal N.
HARD_TIER_SOURCE = "AIME"
HARD_TIER_BALANCED = False

# Easy/Medium remain Hendrycks-sourced (unchanged from the prior design).
EASY_MEDIUM_PER_CATEGORY = 10

DATASET_SEED = 25116096  # unchanged from the prior pipeline — keep for continuity

# ==========================================================
# MODELS
#
# All three models are free-tier, on the three providers already integrated.
# Gemini is fixed per supervisor requirement. Groq and Mistral were reselected
# after review found the prior Llama-3.3-70B and Mistral-Small were weak on
# AIME-level problems, and after confirming Llama-4-Maverick/Scout (the
# originally planned Groq replacement) were deprecated by Groq in Feb/Jun 2026.
#
# Both Gemini and Groq's replacement model expose a "thinking effort" control.
# It is pinned to its lowest setting and held CONSTANT across CoT and ToT so
# the prompting-strategy manipulation (RQ3), not a variable internal reasoning
# budget, is what's being measured. This is a controlled confound, not an
# eliminated one — documented explicitly in docs/limitations.md.
# ==========================================================

MODELS = {

    "gemini": {
        "provider": "gemini",
        # "gemini-3-flash" (the model ID initially planned from web research)
        # does not exist on this API key's endpoint (404) -- verified via
        # client.models.list() during the pilot. gemini-3.5-flash was tried
        # next but its free tier caps at only 20 requests/DAY (confirmed via
        # the 429 error's quotaValue), making the ~655-call full study take
        # ~33 days for Gemini alone. gemini-flash-lite-latest is used
        # instead -- a newer lite-tier model than the prior pipeline's
        # gemini-3.1-flash-lite, with a much more generous free daily quota
        # (no limit hit in testing). Likely still an improvement over the
        # old flash-lite, though probably weaker than 3.5-flash -- this is
        # an accuracy-for-feasibility tradeoff, not a strictly better pick.
        "model_id": "gemini-flash-lite-latest",
        "temperature": 0,
        "thinking_level": "minimal",  # pinned constant across CoT/ToT
        "max_tokens": 8192,
    },

    "groq": {
        "provider": "groq",
        # Groq's own recommended migration target for the deprecated
        # Llama 4 Maverick/Scout models (see docs/model_pilot_results.md).
        "model_id": "openai/gpt-oss-120b",
        "temperature": 0,
        "reasoning_effort": "low",  # pinned constant across CoT/ToT
        # Groq's free tier caps openai/gpt-oss-120b at 8000 tokens/minute
        # PER REQUEST (prompt + completion combined) -- discovered via a 413
        # during the pilot ("Requested 8403" against a limit of 8000 with
        # max_tokens=8192 alone). 4096 leaves headroom for the prompt and
        # for reasoning tokens consumed internally even at reasoning_effort="low".
        "max_tokens": 4096,
    },

    "mistral": {
        "provider": "mistral",
        # Free on Mistral's "Experiment" tier. Standard instruct model,
        # no thinking-mode toggle to control for.
        "model_id": "mistral-large-latest",
        "temperature": 0,
        "max_tokens": 8192,
    },

}

MODEL_NAMES = list(MODELS.keys())

# ==========================================================
# PROMPTS
#
# Every result row records which prompt *version* produced it (see
# src/utils/io.py), so a prompt-wording tweak mid-project is distinguishable
# from the original run rather than silently blending both into one column.
# ==========================================================

PROMPT_TYPES = ["cot", "tot"]

PROMPT_VERSIONS = {
    "cot": "cot_v1",
    "tot": "tot_v1",
}

# Real multi-branch Tree-of-Thought (not the single-call approximation from
# the prior pipeline). Each ToT question costs TOT_BRANCH_COUNT generation
# calls + 1 selection call, vs. 1 call for CoT.
TOT_BRANCH_COUNT = 3

# All three models are pinned to temperature=0 for CoT and for the ToT
# selection call, so those steps are deterministic and comparable. Branch
# GENERATION is the one exception: at temperature=0, calling the same branch
# prompt N times returns near-identical greedy-decoded text each time, which
# would make "N branches" meaningless — there would be nothing to search
# over. TOT_BRANCH_TEMPERATURE introduces just enough sampling diversity for
# the branches to actually differ, matching standard practice in
# self-consistency / Tree-of-Thought implementations. reasoning_effort
# (Groq/GPT-OSS) stays pinned at its configured value throughout — it
# controls internal reasoning depth, a different axis from sampling
# temperature.
TOT_BRANCH_TEMPERATURE = 0.7

# ==========================================================
# ERROR TAXONOMY / JUDGE
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
    "Other",
]

# Primary judge stays Gemini (cost reasons). A validation sample is re-judged
# by a neutral model not under evaluation, and inter-judge agreement (Cohen's
# kappa) is computed and stored in outputs/stats/ — see src/evaluation/error_judge.py.
PRIMARY_JUDGE_MODEL = "gemini"
NEUTRAL_JUDGE_MODEL_ID = "llama-3.3-70b-versatile"  # not gemini, not under evaluation
# Originally Kimi K2 (moonshotai/kimi-k2-instruct-0905) -- removed from Groq's
# model catalog entirely between setup and running this validation pass
# (confirmed via a live client.models.list() call, 404 on every request).
# llama-3.3-70b-versatile is the strongest remaining Groq model that isn't
# one of the three study models (Meta, not OpenAI/Gemini/Mistral), keeping
# judge independence intact.
VALIDATION_SAMPLE_SIZE = 120

# ==========================================================
# RETRY / RATE-LIMIT SETTINGS
#
# Providers' free tiers differ substantially — Mistral's Experiment tier is
# 2 requests/minute, which is a scheduling fact, not a bug, and the resumable
# runner must pace accordingly rather than hammering into 429s.
# ==========================================================

RETRY_SETTINGS = {

    "gemini": {"max_retries": 5, "wait_seconds": 60, "min_request_interval_s": 4.0},

    "groq": {"max_retries": 5, "wait_seconds": 30, "min_request_interval_s": 2.0},

    "mistral": {"max_retries": 5, "wait_seconds": 45, "min_request_interval_s": 31.0},

}
