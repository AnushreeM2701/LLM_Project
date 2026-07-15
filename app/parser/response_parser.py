import re


# ==========================================================
# Extract Final Answer
# ==========================================================

# ==========================================================
# Extract Final Answer
# ==========================================================

def extract_final_answer(response):
    """
    Extract the model's final answer.

    Priority:
    1. \\boxed{...}
    2. Final answer:
    3. Answer:
    4. Last non-empty line
    """

    response = str(response).strip()

    if not response:
        return ""

    # ------------------------------------------------------
    # 1. Look for \boxed{...}
    # ------------------------------------------------------

    start = response.rfind(r"\boxed{")

    if start != -1:

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

    # ------------------------------------------------------
    # 2. Final answer
    # ------------------------------------------------------

    match = re.search(

        r"final\s+answer\s+(?:is\s+)?[:\-]?\s*\$?(.+?)\$?(?:\.|\n|$)",

        response,

        flags=re.IGNORECASE

    )

    if match:

        answer = match.group(1).strip()

        answer = answer.strip("$")

        answer = answer.rstrip(".")

        return answer

    # ------------------------------------------------------
    # 3. Answer:
    # ------------------------------------------------------

    match = re.search(

        r"answer\s*[:\-]?\s*(.+)",

        response,

        flags=re.IGNORECASE

    )

    if match:

        return match.group(1).strip().strip("$").strip(".")

    # ------------------------------------------------------
    # 4. Last non-empty line
    # ------------------------------------------------------

    lines = [

        line.strip()

        for line in response.splitlines()

        if line.strip()

    ]

    if lines:

        return lines[-1].strip().strip("$").strip(".")

    return ""


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