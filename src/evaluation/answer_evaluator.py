import math
import re
from fractions import Fraction

from sympy import simplify, sympify  # noqa: F401 (sympify kept for parity/back-compat)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

transformations = (
    standard_transformations
    + (implicit_multiplication_application,)
)

def normalize_answer(answer):

    if answer is None:
        return ""

    answer = str(answer).strip().lower()

    for token in ["```", "**", "__"]:
        answer = answer.replace(token, "")

    answer = answer.replace(",", "")
    answer = answer.replace("$", "")

    answer = answer.replace(r"\(", "")
    answer = answer.replace(r"\)", "")
    # LaTeX escapes a literal percent sign as "\%" (plain "%" starts a
    # comment) -- left as-is, "10\%" fails the percent-fullmatch check below
    # (which requires digits immediately followed by "%") and never gets
    # converted to 0.1, so it stops comparing equal to a plain "10%".
    answer = answer.replace(r"\%", "%")

    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", answer)

    answer = re.sub(
        r"\\d?frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}",
        r"\1/\2",
        answer,
    )
    # Brace-less LaTeX shorthand, e.g. "\frac12" == "\frac{1}{2}" -- see
    # src/parser/response_parser.py's clean_answer() for the same fix and
    # why it matters (unconverted, adjacent digits misread as one number).
    answer = re.sub(r"\\d?frac(\w)(\w)", r"\1/\2", answer)

    answer = answer.replace("^", "**")

    answer = re.sub(r"\s+", "", answer)

    m = re.fullmatch(r"(-?\d+(\.\d+)?)%", answer)
    if m:
        answer = str(float(m.group(1)) / 100)

    return answer


def to_number(value):

    # Mixed numbers ("2 1/3") must be checked BEFORE normalize_answer, which
    # strips all whitespace and would otherwise collapse "2 1/3" into "21/3"
    # — silently misparsed as the plain fraction 21/3 instead of 2 + 1/3.
    # (This was dead code in the prior evaluator for exactly this reason.)
    if value is not None:

        raw = str(value).strip().lower().replace(",", "").replace("$", "")

        m = re.fullmatch(r"(-?)(\d+)\s+(\d+)/(\d+)", raw)

        if m:
            try:
                sign_str, whole, num, den = m.groups()
                sign = -1 if sign_str == "-" else 1
                magnitude = int(whole) + Fraction(int(num), int(den))
                return float(sign * magnitude)
            except (ZeroDivisionError, ValueError):
                pass

    value = normalize_answer(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    if re.fullmatch(r"-?\d+/\d+", value):
        try:
            return float(Fraction(value))
        except (ZeroDivisionError, ValueError):
            pass

    return None


def compare_numeric(gt, pred):

    gt_num = to_number(gt)
    pred_num = to_number(pred)

    if gt_num is None or pred_num is None:
        return False

    return math.isclose(gt_num, pred_num, rel_tol=1e-9, abs_tol=1e-9)


def compare_symbolic(gt, pred):

    gt = normalize_answer(gt)
    pred = normalize_answer(pred)

    try:

        gt_expr = parse_expr(gt, transformations=transformations, evaluate=True)
        pred_expr = parse_expr(pred, transformations=transformations, evaluate=True)

        return simplify(gt_expr - pred_expr) == 0

    except Exception:
        return False


def compare_text(gt, pred):

    return normalize_answer(gt) == normalize_answer(pred)


def is_correct(gt, pred):

    if compare_numeric(gt, pred):
        return True

    if compare_symbolic(gt, pred):
        return True

    if compare_text(gt, pred):
        return True

    return False


def evaluate_response(ground_truth, prediction):

    gt = normalize_answer(ground_truth)
    pred = normalize_answer(prediction)

    return {
        "ground_truth": gt,
        "prediction": pred,
        "correct": is_correct(gt, pred),
    }
