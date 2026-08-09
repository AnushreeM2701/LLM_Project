import pytest

from src.evaluation.answer_evaluator import evaluate_response, is_correct

# CASES CARRIED OVER FROM THE PRIOR PIPELINE'S SMOKE TEST
@pytest.mark.parametrize(
    "ground_truth,prediction",
    [
        (r"\frac{2}{9}", "2/9"),
        (r"\dfrac{2}{9}", "2/9"),
        ("0.5", "50%"),
        ("24x^2-6x+3", "3*(8*x^2-2*x+1)"),
        ("x^2+2x+1", "(x+1)^2"),
        ("true", "TRUE"),
        ("1000", "1,000"),
    ],
)
def test_known_equivalent_pairs(ground_truth, prediction):
    assert is_correct(ground_truth, prediction)

# NEW EDGE CASES

def test_mixed_number_equivalence():
    assert is_correct("2 1/3", "7/3")


def test_negative_fraction_equivalence():
    assert is_correct("-3/4", "-0.75")


def test_boxed_latex_answer():
    assert is_correct(r"\boxed{42}", "42")


def test_percentage_vs_decimal():
    assert is_correct("25%", "0.25")


def test_clearly_wrong_numeric_answer():
    assert not is_correct("5", "6")


def test_clearly_wrong_symbolic_answer():
    assert not is_correct("x^2 + 1", "x^2 - 1")


def test_empty_prediction_is_not_correct():
    assert not is_correct("42", "")


def test_none_prediction_is_not_correct():
    assert not is_correct("42", None)


def test_aime_style_integer_answer():
    # AIME answers are always integers 0-999.
    assert is_correct("204", "204")
    assert not is_correct("204", "205")


def test_evaluate_response_returns_expected_shape():
    result = evaluate_response("1/2", "0.5")
    assert set(result.keys()) == {"ground_truth", "prediction", "correct"}
    assert result["correct"] is True


def test_escaped_percent_sign_equivalence():
    assert is_correct(r"10\%", "10%")


def test_braceless_frac_shorthand_equivalence():
    assert is_correct(r"\frac12", "1/2")
