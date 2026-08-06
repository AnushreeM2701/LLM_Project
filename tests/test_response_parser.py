from src.parser.response_parser import parse_response


def test_extract_final_answer_ignores_embedded_ordinal():
    # Regression: "The 158th marble is gray." used to extract "158" (the
    # ordinal reference) instead of "gray" (the actual answer), because the
    # numeric-pattern extraction matched "158" before the word fallback ever
    # ran. Found via NT_E_009_groq_cot in the live results data.
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
    # An ordinal elsewhere in the sentence must not swallow a genuinely
    # numeric final answer that appears after it.
    response = "Final Answer: On the 3rd attempt, the result is 17."
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "17"


def test_extract_final_answer_handles_dfrac():
    # Regression: "\dfrac{3}{16}" used to extract "3" instead of "3/16",
    # because clean_answer() only converted the plain "\frac" macro, not the
    # "\dfrac" (display-style) variant -- the numeric fallback then grabbed
    # the stray "3" from inside the unconverted macro. Found via
    # PROB_E_003_groq_tot in the live results data.
    response = "Final Answer: \\(\\dfrac{3}{16}\\)"
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "3/16"


def test_extract_final_answer_handles_tfrac():
    response = "Final Answer: \\tfrac{2}{7}"
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "2/7"


def test_extract_final_answer_ignores_inline_exponent():
    # Regression: "n^2 has 5 divisors" used to extract "2" (the exponent in
    # "n^2") instead of "5" (the actual answer). Found via
    # NT_E_007_mistral_tot in the live results data.
    response = "Final Answer: \\( n^2 \\) has **5** divisors."
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "5"


def test_extract_final_answer_handles_braceless_frac_shorthand():
    # Regression: "\frac12" (LaTeX's brace-less shorthand for single-char
    # numerator/denominator, == "\frac{1}{2}") wasn't recognized by the
    # brace-requiring frac regex, so the digit-extraction fallback read the
    # adjacent "1" and "2" as one contiguous number "12" instead of a
    # fraction. Found via ALG_M_008_groq_cot in the live results data.
    response = "**Final Answer:** \\(\\displaystyle \\frac12\\)"
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "1/2"


def test_extract_final_answer_handles_function_call_assignment():
    # Regression: "f(5) = 7" wasn't recognized by the "strip leading
    # variable-assignment prefix" regex (which only handled plain "x = 5"),
    # so the numeric fallback grabbed "5" (the function's argument) instead
    # of "7" (the actual value after "="). Found via ALG_M_010_mistral_cot
    # in the live results data.
    response = "Final Answer: \\( f(5) = 7 \\)"
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "7"


def test_extract_final_answer_prefers_trailing_is_clause():
    # Regression: "...numbers that sum to 87 is 3" used to extract "87"
    # (the restated problem parameter) instead of "3" (the actual answer,
    # at the end after "is"). Found via NT_M_001_mistral_tot in the live
    # results data.
    response = (
        "### Final Answer\n"
        "The minimum positive difference between two composite numbers "
        "that sum to 87 is **3**."
    )
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "3"


def test_extract_boxed_handles_dfrac():
    # Regression: "\boxed{\dfrac{25}{84}}" matched neither of
    # extract_boxed's regexes (both assumed literal "\frac"), so it fell
    # through to the "Final Answer" heading-then-blank-line fallback and
    # grabbed the trailing word "is" from unrelated prose several lines
    # later ("...the solution to the equation is:"). Found via
    # ALG_M_003_mistral_cot in the live results data.
    response = (
        "### Final Answer\n\n"
        "After breaking it down, the solution to the equation is:\n\n"
        "\\boxed{\\dfrac{25}{84}}"
    )
    parsed = parse_response(response)
    assert parsed["model_final_answer"] == "25/84"
