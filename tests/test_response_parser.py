from src.parser.response_parser import parse_response


def test_extract_final_answer_ignores_embedded_ordinal():
    response = (
        "Step 1: Identify the repeating pattern.\n\n"
        "**Final Answer:** The 158th marble is **gray**."
    )
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "gray"


def test_extract_final_answer_still_prefers_numeric_when_no_ordinal():
    response = "Final Answer: 42"
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "42"


def test_extract_final_answer_numeric_answer_with_unrelated_ordinal_context():
    response = "Final Answer: On the 3rd attempt, the result is 17."
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "17"


def test_extract_final_answer_handles_dfrac():
    response = "Final Answer: \\(\\dfrac{3}{16}\\)"
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "3/16"


def test_extract_final_answer_handles_tfrac():
    response = "Final Answer: \\tfrac{2}{7}"
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "2/7"


def test_extract_final_answer_ignores_inline_exponent():
    response = "Final Answer: \\( n^2 \\) has **5** divisors."
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "5"


def test_extract_final_answer_handles_braceless_frac_shorthand():
    response = "**Final Answer:** \\(\\displaystyle \\frac12\\)"
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "1/2"


def test_extract_final_answer_handles_function_call_assignment():
    response = "Final Answer: \\( f(5) = 7 \\)"
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "7"


def test_extract_final_answer_prefers_trailing_is_clause():
    response = (
        "### Final Answer\n"
        "The minimum positive difference between two composite numbers "
        "that sum to 87 is **3**."
    )
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "3"


def test_extract_boxed_handles_dfrac():
    response = (
        "### Final Answer\n\n"
        "After breaking it down, the solution to the equation is:\n\n"
        "\\boxed{\\dfrac{25}{84}}"
    )
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "25/84"
