import re


def extract_final_answer_and_match(response):
    """
    Extract the final answer from the model response.

    Returns:
        (answer_str, match_span_start, match_span_end)
        If no answer is found, returns ("", None, None).

    Note: match spans are used to better trim reasoning.
    """

    patterns = [
        # \boxed{...}
        (r"\\boxed\{(.+?)\}", 0),
        # Final Answer: ...
        (r"Final Answer\s*[:\-]\s*(.+)", 1),
        # The final answer is ...
        (r"The final answer is\s*(.+)", 1),
        # Answer: ...
        (r"Answer\s*[:\-]\s*(.+)", 1),
    ]

    for pattern, _group in patterns:
        match = re.search(
            pattern,
            response,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            answer = match.group(1).strip() if match.lastindex else ""
            return answer, match.start(1), match.end(1)

    return "", None, None


def extract_final_answer(response):
    """Extract final answer string only."""
    answer, _, _ = extract_final_answer_and_match(response)
    return answer


def extract_reasoning(response):
    """Extract reasoning by trimming away the detected final-answer region when possible."""

    answer, a_start, a_end = extract_final_answer_and_match(response)

    if a_start is not None and a_end is not None:
        # Keep everything before the final answer substring.
        reasoning = response[:a_start]
        return reasoning.strip()

    # Fallback: heuristic removal
    reasoning = re.sub(
        r"(Final Answer|The final answer is|Answer\s*:).*",
        "",
        response,
        flags=re.IGNORECASE | re.DOTALL
    )

    return reasoning.strip()



def count_reasoning_steps(reasoning):

    """
    Placeholder implementation.

    Later this will be replaced by
    an LLM-based reasoning step counter.
    """

    lines = []

    for line in reasoning.split("\n"):

        line = line.strip()

        if line:
            lines.append(line)

    return len(lines)


def parse_response(response):
    """
    Parse the complete model response.
    """

    reasoning = extract_reasoning(response)

    answer = extract_final_answer(response)

    steps = count_reasoning_steps(reasoning)

    return {

        "full_response": response,

        "reasoning": reasoning,

        "final_answer": answer,

        "step_count": steps

    }


if __name__ == "__main__":

    sample_response = """
A fair coin has two equally likely outcomes.

Heads

Tails

Probability = 1/2

Therefore,

The final answer is 1/2.
"""

    result = parse_response(sample_response)

    print(result)