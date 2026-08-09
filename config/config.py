import os

# PATHS
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

# DATASET COMPOSITION

CATEGORIES = ["Algebra", "Number Theory", "Probability", "Combinatorics"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]

HARD_TIER_SOURCE = "AIME"
HARD_TIER_BALANCED = False

# Easy/Medium remain Hendrycks-sourced 
EASY_MEDIUM_PER_CATEGORY = 10

DATASET_SEED = 25116096  

MODELS = {

    "gemini": {
        "provider": "gemini",
        # "gemini-3-flash" 
        "model_id": "gemini-flash-lite-latest",
        "temperature": 0,
        "thinking_level": "minimal",  # pinned constant across CoT/ToT
        "max_tokens": 8192,
    },

    "groq": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "temperature": 0,
        "reasoning_effort": "low",
        "max_tokens": 4096,
    },

    "mistral": {
        "provider": "mistral",
        "model_id": "mistral-large-latest",
        "temperature": 0,
        "max_tokens": 8192,
    },

}

MODEL_NAMES = list(MODELS.keys())

PROMPT_TYPES = ["cot", "tot"]

PROMPT_VERSIONS = {
    "cot": "cot_v1",
    "tot": "tot_v1",
}

TOT_BRANCH_COUNT = 3

TOT_BRANCH_TEMPERATURE = 0.7

# ERROR TAXONOMY / JUDGE

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

PRIMARY_JUDGE_MODEL = "gemini"
NEUTRAL_JUDGE_MODEL_ID = "llama-3.3-70b-versatile"

VALIDATION_SAMPLE_SIZE = 120

RETRY_SETTINGS = {

    "gemini": {"max_retries": 5, "wait_seconds": 60, "min_request_interval_s": 4.0},

    "groq": {"max_retries": 5, "wait_seconds": 30, "min_request_interval_s": 2.0},

    "mistral": {"max_retries": 5, "wait_seconds": 45, "min_request_interval_s": 31.0},

}
