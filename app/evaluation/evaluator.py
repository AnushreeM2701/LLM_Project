"""
evaluator.py

Evaluation utilities for comparing model answers with ground truth.
Supports:
- Integers
- Decimals
- Fractions
- Percentages
- Mixed fractions
- Algebraic expressions (SymPy)
- Text answers
"""

import math
import re
from fractions import Fraction

from sympy import simplify, sympify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

transformations = (
    standard_transformations
    + (implicit_multiplication_application,)
)


# ==========================================================
# NORMALIZE ANSWER
# ==========================================================

def normalize_answer(answer):

    if answer is None:
        return ""

    answer = str(answer).strip().lower()

    # Remove markdown
    for token in ["```", "**", "__"]:
        answer = answer.replace(token, "")

    answer = answer.replace(",", "")
    answer = answer.replace("$", "")

    answer = answer.replace(r"\(", "")
    answer = answer.replace(r"\)", "")

    # Remove LaTeX wrappers
    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", answer)

    # Handle both \frac and \dfrac
    answer = re.sub(
        r"\\d?frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}",
        r"\1/\2",
        answer,
    )

    # Convert ^ to **
    answer = answer.replace("^", "**")

    # Remove spaces
    answer = re.sub(r"\s+", "", answer)

    # Convert percentage
    m = re.fullmatch(r"(-?\d+(\.\d+)?)%", answer)
    if m:
        answer = str(float(m.group(1)) / 100)

    return answer


# ==========================================================
# NUMBER CONVERSION
# ==========================================================

def to_number(value):

    value = normalize_answer(value)

    try:
        return float(value)
    except:
        pass

    if re.fullmatch(r"-?\d+/\d+", value):
        try:
            return float(Fraction(value))
        except:
            pass

    if re.fullmatch(r"-?\d+\s+\d+/\d+", value):
        try:
            whole, frac = value.split()
            return float(int(whole) + Fraction(frac))
        except:
            pass

    return None


# ==========================================================
# NUMERIC COMPARISON
# ==========================================================

def compare_numeric(gt, pred):

    gt_num = to_number(gt)
    pred_num = to_number(pred)

    if gt_num is None or pred_num is None:
        return False

    return math.isclose(
        gt_num,
        pred_num,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


# ==========================================================
# ALGEBRAIC COMPARISON
# ==========================================================

def compare_symbolic(gt, pred):

    gt = normalize_answer(gt)
    pred = normalize_answer(pred)

    try:

        gt_expr = parse_expr(
            gt,
            transformations=transformations,
            evaluate=True,
        )

        pred_expr = parse_expr(
            pred,
            transformations=transformations,
            evaluate=True,
        )

        return simplify(gt_expr - pred_expr) == 0

    except Exception:
        return False


# ==========================================================
# TEXT COMPARISON
# ==========================================================

def compare_text(gt, pred):

    return normalize_answer(gt) == normalize_answer(pred)


# ==========================================================
# MAIN COMPARISON
# ==========================================================

def is_correct(gt, pred):

    if compare_numeric(gt, pred):
        return True

    if compare_symbolic(gt, pred):
        return True

    if compare_text(gt, pred):
        return True

    return False


# ==========================================================
# EVALUATE RESPONSE
# ==========================================================

def evaluate_response(ground_truth, prediction):

    gt = normalize_answer(ground_truth)
    pred = normalize_answer(prediction)

    return {

        "ground_truth": gt,

        "prediction": pred,

        "correct": is_correct(gt, pred),

    }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    tests = [

        (r"\frac{2}{9}", "2/9"),
        (r"\dfrac{2}{9}", "2/9"),
        ("0.5", "50%"),
        ("24x^2-6x+3", "3*(8*x^2-2*x+1)"),
        ("x^2+2x+1", "(x+1)^2"),
        ("true", "TRUE"),
        ("1000", "1,000"),
    ]

    print("=" * 60)
    print("Evaluator Test")
    print("=" * 60)

    for gt, pred in tests:

        result = evaluate_response(gt, pred)

        print()

        print("Ground Truth :", gt)
        print("Prediction   :", pred)
        print("Correct      :", result["correct"])