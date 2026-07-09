import re


# ==========================================================
# Extract Final Answer
# ==========================================================

def extract_final_answer(response):
    """
    Extract the answer inside \\boxed{...}.
    Handles nested braces correctly.
    """

    response = str(response)

    start = response.rfind(r"\boxed{")

    if start == -1:
        return ""

    i = start + len(r"\boxed{")

    brace_count = 1

    answer = ""

    while i < len(response):

        char = response[i]

        if char == "{":
            brace_count += 1

        elif char == "}":
            brace_count -= 1

            if brace_count == 0:
                break

        answer += char

        i += 1

    return answer.strip()


# ==========================================================
# Count Reasoning Steps
# ==========================================================

def count_steps(response):
    """
    Estimate the number of reasoning steps.

    Current strategy:
    Count numbered steps if present.
    Otherwise count non-empty lines.

    This will later be replaced by an LLM-based
    step counter.
    """

    response = str(response)

    numbered = re.findall(
        r'^\s*\d+[\.\)]',
        response,
        flags=re.MULTILINE
    )

    if len(numbered) > 0:
        return len(numbered)

    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    return len(lines)


# ==========================================================
# Parse Response
# ==========================================================

def parse_response(response):

    return {

        "model_response": response,

        "model_final_answer": extract_final_answer(response),

        "model_step_count": count_steps(response)

    }