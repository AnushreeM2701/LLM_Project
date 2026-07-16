import re
from fractions import Fraction


# ==========================================================
# Normalize Answers
# ==========================================================

def normalize_answer(answer):

    if answer is None:
        return ""

    answer = str(answer).strip()

    # ------------------------------------------
    # Remove markdown
    # ------------------------------------------

    answer = answer.replace("**", "")
    answer = answer.replace("__", "")
    answer = answer.replace("`", "")

    # ------------------------------------------
    # Remove LaTeX wrappers
    # ------------------------------------------

    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("$", "")
    answer = answer.replace("\\(", "")
    answer = answer.replace("\\)", "")

    # ------------------------------------------
    # \boxed{}
    # ------------------------------------------

    answer = re.sub(
        r"\\boxed\{([^}]*)\}",
        r"\1",
        answer
    )

    # ------------------------------------------
    # \text{Gray}
    # ------------------------------------------

    answer = re.sub(
        r"\\text\{([^}]*)\}",
        r"\1",
        answer
    )

    # ------------------------------------------
    # \frac{a}{b}
    # ------------------------------------------

    answer = re.sub(
        r"\\d?frac\{([^}]*)\}\{([^}]*)\}",
        r"\1/\2",
        answer
    )

    # ------------------------------------------
    # x=5 , m =2
    # ------------------------------------------

    answer = re.sub(
        r"^[A-Za-z]+\s*=\s*",
        "",
        answer
    )

    # ------------------------------------------
    # Remove common words
    # ------------------------------------------

    answer = re.sub(
        r"^(the\s+)?(final\s+)?answer\s*(is)?\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )

    # ------------------------------------------
    # Remove units
    # ------------------------------------------

    answer = re.sub(
        r"\b(minutes?|minute|hours?|hour|seconds?|km|km/h|meters?)\b",
        "",
        answer,
        flags=re.IGNORECASE
    )

    # ------------------------------------------
    # Remove punctuation
    # ------------------------------------------

    answer = answer.strip()

    answer = answer.rstrip(".")

    answer = answer.rstrip(",")

    # ------------------------------------------
    # Remove ALL spaces
    # ------------------------------------------

    answer = re.sub(r"\s+", "", answer)
    answer = answer.lower()

    return answer


# ==========================================================
# Compare Answers
# ==========================================================

def compare_answers(ground_truth, model_answer):

    gt = normalize_answer(ground_truth)

    pred = normalize_answer(model_answer)

    # ------------------------------------------
    # Exact match
    # ------------------------------------------

    if gt == pred:

        return True

    # ------------------------------------------
    # Fractions
    # ------------------------------------------

    try:

        if Fraction(gt) == Fraction(pred):

            return True

    except Exception:

        pass

    # ------------------------------------------
    # Floats
    # ------------------------------------------

    try:

        if abs(float(gt) - float(pred)) < 1e-8:

            return True

    except Exception:

        pass

    return False


# ==========================================================
# Evaluate
# ==========================================================

def evaluate_response(ground_truth, model_answer):

    correct = compare_answers(
        ground_truth,
        model_answer
    )

    return {

        "answer_correct": correct,

        "error_type": "" if correct else "Incorrect Answer"

    }