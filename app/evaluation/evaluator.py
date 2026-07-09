import re
from fractions import Fraction


# ==========================================================
# Normalize Answers
# ==========================================================

def normalize_answer(answer):
    """
    Normalize answers so equivalent answers
    can be compared.

    Examples:
    \\frac{1}{2} -> 1/2
    \\dfrac{3}{4} -> 3/4
    """

    if answer is None:
        return ""

    answer = str(answer).strip()

    # Remove spaces
    answer = answer.replace(" ", "")

    # Convert LaTeX fractions
    answer = re.sub(
        r"\\d?frac\{(.*?)\}\{(.*?)\}",
        r"\1/\2",
        answer
    )

    # Remove LaTeX wrappers
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("$", "")

    return answer


# ==========================================================
# Compare Answers
# ==========================================================

def compare_answers(ground_truth, model_answer):
    """
    Return True if answers are equivalent.
    """

    gt = normalize_answer(ground_truth)
    pred = normalize_answer(model_answer)

    # Exact match
    if gt == pred:
        return True

    # Try fraction comparison
    try:

        gt_fraction = Fraction(gt)
        pred_fraction = Fraction(pred)

        return gt_fraction == pred_fraction

    except Exception:

        pass

    # Try float comparison
    try:

        return abs(float(gt) - float(pred)) < 1e-8

    except Exception:

        pass

    return False


# ==========================================================
# Evaluate
# ==========================================================

def evaluate_response(ground_truth, model_answer):

    return {

        "answer_correct": compare_answers(
            ground_truth,
            model_answer
        ),

        "error_type": ""

    }