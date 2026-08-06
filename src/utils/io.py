"""
Canonical read/write layer for the experiment results file.

Every other module reaches the results/dataset ONLY through this module's
functions — never by hardcoding a path. This is what makes the "stale copy"
class of bug (two analysis scripts silently reading different files) and the
"repair script rewrote the file wrong" class of bug structurally impossible:
there is exactly one path, exactly one writer, and one place that owns the
invariant "Answer Correct and Error Type must never disagree."
"""

import os
import tempfile
import threading

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment

from config.config import (
    RESULTS_CSV_PATH,
    RESULTS_XLSX_PATH,
    FINAL_DATASET_PATH,
)

# ==========================================================
# RESULT SCHEMA
# ==========================================================

RESULT_COLUMNS = [
    "Experiment ID",
    "Question Number",
    "Question ID",
    "Category",
    "Question",
    "Difficulty",
    "Model",
    "Model Version",       # provenance: exact served model version/string
    "Prompt",
    "Prompt Version",      # provenance: config.PROMPT_VERSIONS value
    "Ground Truth Solution",
    "Ground Truth Answer",
    "Model Response",
    "Model Final Answer",
    "Reasoning",
    "Step Count",
    "Answer Correct",
    "Start Time",
    "End Time",
    "Execution Time (s)",
    "Error Step",
    "Error Type",
    "Error Subtype",       # free-text specific error pattern, alongside the fixed Error Type category
    "Confidence",
    "Justification",
    "Judge Model",         # provenance: which model produced the error classification
]

# Columns that depend on the CURRENT value of "Answer Correct". Any script
# that changes "Answer Correct" for a row must clear these so the row is
# re-picked-up by the error judge rather than left with a stale label from
# before the change. This is the direct fix for the 6-row inconsistency bug
# found in the prior pipeline (Answer Correct=False but Error Type="Correct").
DEPENDENT_ON_CORRECTNESS = [
    "Error Step",
    "Error Type",
    "Error Subtype",
    "Confidence",
    "Justification",
    "Judge Model",
]


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write via a temp file + os.replace so a crash mid-write can never
    leave a truncated/corrupted canonical file on disk."""

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")

    try:
        with os.fdopen(fd, "w", newline="") as f:
            df.to_csv(f, index=False)

        os.replace(tmp_path, path)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _strip_illegal_excel_chars(df: pd.DataFrame) -> pd.DataFrame:
    """XML 1.0 (the format underlying .xlsx) forbids most control characters
    -- openpyxl raises IllegalCharacterError rather than silently dropping
    them. LLM-generated text (e.g. a judge's Justification) occasionally
    contains one, which would otherwise crash every subsequent save_results()
    call for the rest of the run. The canonical CSV is unaffected -- this
    only sanitizes the Excel mirror."""

    def clean(value):
        if isinstance(value, str):
            return ILLEGAL_CHARACTERS_RE.sub("", value)
        return value

    return df.map(clean)


def _write_formatted_excel(df: pd.DataFrame, path: str, sheet_name: str) -> None:

    # openpyxl/pandas dispatch on file extension, so the temp file must
    # still end in .xlsx (a plain ".tmp" suffix raises "Invalid extension").
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".xlsx")
    os.close(fd)

    df = _strip_illegal_excel_chars(df)

    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:

        df.to_excel(writer, index=False, sheet_name=sheet_name)

        worksheet = writer.sheets[sheet_name]

        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for row in worksheet.iter_rows():

            max_lines = 1

            for cell in row:

                cell.alignment = Alignment(wrap_text=True, vertical="top")

                if cell.value:
                    max_lines = max(max_lines, str(cell.value).count("\n") + 1)

            worksheet.row_dimensions[row[0].row].height = max(25, max_lines * 18)

    os.replace(tmp_path, path)


# ==========================================================
# RESULTS: LOAD / SAVE
# ==========================================================

# Numeric/boolean columns are left as real NaN (downstream code uses
# pd.to_numeric(..., errors="coerce") for these) -- everything else is
# text that should read back as "" when blank, never NaN.
_NON_TEXT_COLUMNS = {"Question Number", "Step Count", "Execution Time (s)", "Answer Correct"}
_TEXT_COLUMNS = [c for c in RESULT_COLUMNS if c not in _NON_TEXT_COLUMNS]


def load_results() -> pd.DataFrame:
    """Loads the canonical results file. Blank text cells come back as ""
    (never NaN) -- a CSV round-trip otherwise turns an untouched cell (e.g.
    Error Type before judging) into a float NaN, and str(NaN) == "nan", NOT
    "" -- every "is this field blank?" check elsewhere (error_judge.py,
    the analysis modules) silently misfires without this normalization."""

    if not os.path.exists(RESULTS_CSV_PATH):
        return pd.DataFrame(columns=RESULT_COLUMNS)

    df = pd.read_csv(RESULTS_CSV_PATH)

    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df[_TEXT_COLUMNS] = df[_TEXT_COLUMNS].fillna("")

    return df[RESULT_COLUMNS]


def save_results(df: pd.DataFrame) -> None:
    """Full-file rewrite: atomic CSV write + regenerated formatted Excel.

    Use this (not raw pandas) whenever a script rewrites the whole results
    table (e.g. re-evaluation, error judging) rather than appending one row.
    """

    assert_no_stale_error_labels(df)

    df = df[RESULT_COLUMNS]

    _atomic_write_csv(df, RESULTS_CSV_PATH)
    _write_formatted_excel(df, RESULTS_XLSX_PATH, sheet_name="Experiment Results")


# Guards concurrent appends when models run in parallel threads (see
# src/experiments/run_experiment.py) -- without this, two threads racing to
# check "does the file exist" / append a row can corrupt the CSV exactly
# like the two-process race that corrupted an earlier pilot run.
_append_lock = threading.Lock()


def append_result(result: dict) -> None:
    """Append a single freshly-generated experiment row to the canonical
    results file. Used by the experiment runner during data collection.
    Thread-safe -- serializes concurrent appends from parallel model runs."""

    for col in RESULT_COLUMNS:
        result.setdefault(col, "")

    row_df = pd.DataFrame([result])[RESULT_COLUMNS]

    with _append_lock:
        if os.path.exists(RESULTS_CSV_PATH):
            row_df.to_csv(RESULTS_CSV_PATH, mode="a", header=False, index=False)
        else:
            os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
            row_df.to_csv(RESULTS_CSV_PATH, mode="w", header=True, index=False)


def load_completed_experiments() -> set:
    """(Question ID, Model, Prompt) tuples already collected — for resuming
    an interrupted run without re-spending API calls."""

    if not os.path.exists(RESULTS_CSV_PATH):
        return set()

    df = pd.read_csv(RESULTS_CSV_PATH)

    return set(
        zip(df["Question ID"], df["Model"], df["Prompt"])
    )


# ==========================================================
# CORRECTNESS INVARIANT
# ==========================================================

def update_correctness(df: pd.DataFrame, index, new_correct: bool) -> pd.DataFrame:
    """Update Answer Correct for a row. If the value actually changed,
    clear the dependent error-classification columns so the row is
    re-judged rather than left carrying a label computed under the old
    correctness value.
    """

    old_correct = df.at[index, "Answer Correct"]

    # Old CSVs may store this as the string "True"/"False" — normalize.
    if isinstance(old_correct, str):
        old_correct = old_correct.strip().lower() == "true"

    df.at[index, "Answer Correct"] = new_correct

    if bool(old_correct) != bool(new_correct):
        for col in DEPENDENT_ON_CORRECTNESS:
            df.at[index, col] = ""

    return df


def assert_no_stale_error_labels(df: pd.DataFrame) -> None:
    """Fail loudly rather than silently persisting the exact inconsistency
    found in the prior pipeline: Answer Correct == False but
    Error Type == "Correct" (or vice versa)."""

    if "Answer Correct" not in df.columns or "Error Type" not in df.columns:
        return

    # fillna("") before stringifying -- an untouched Error Type column
    # round-trips through CSV as NaN (float), and .astype(str) alone turns
    # NaN into the STRING "nan", which then fails the "== ''" check below
    # and gets misdetected as "labeled something other than Correct".
    error_type = df["Error Type"].fillna("").astype(str).str.strip()

    correct = df["Answer Correct"].astype(str).str.strip().str.lower() == "true"
    labeled_correct = error_type == "Correct"
    labeled_something_else = (error_type != "") & ~labeled_correct

    bad = (~correct & labeled_correct) | (correct & labeled_something_else)

    if bad.any():
        bad_ids = df.loc[bad, "Experiment ID"].tolist()
        raise ValueError(
            "Refusing to save: Answer Correct / Error Type disagree for "
            f"{bad.sum()} row(s): {bad_ids[:10]}"
            + (" ..." if bad.sum() > 10 else "")
        )


# ==========================================================
# DATASET
# ==========================================================

def load_dataset() -> pd.DataFrame:
    return pd.read_excel(FINAL_DATASET_PATH)
