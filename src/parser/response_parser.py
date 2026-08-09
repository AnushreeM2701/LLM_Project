import re

# SINGLE-RESPONSE PARSING (ported from the prior pipeline)

def clean_text(text):
    text = str(text)
    for t in ["```", "**", "__", "###", "##", "#"]:
        text = text.replace(t, "")
    return text.strip()


def clean_answer(answer):
    answer = clean_text(answer)
    answer = answer.strip("$").replace(",", "")
    answer = answer.replace(r"\(", "").replace(r"\)", "")
    # See src/evaluation/answer_evaluator.py's normalize_answer() for why
    # "\%" (LaTeX's escaped literal percent sign) needs stripping here too.
    answer = answer.replace(r"\%", "%")
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\boxed\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\[dt]?frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", answer)
    answer = re.sub(r"\\[dt]?frac(\d|[A-Za-z])(\d|[A-Za-z])", r"\1/\2", answer)
    answer = re.sub(r"^\s*[A-Za-z]+(?:\([^()]*\))?\s*=\s*", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip(" .,")


def extract_boxed(response):

    m = re.findall(r"\\boxed\{(\\[dt]?frac\{[^{}]+\}\{[^{}]+\})\}", response)
    if m:
        return clean_answer(m[-1])

    m = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if m:
        return clean_answer(m[-1])

    return None


def extract_expression(text):
    text = clean_answer(text)
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text)
    text = re.sub(r"[A-Za-z]\^\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    is_match = re.search(r"\bis\s*:?\s*(-?\d+(?:\.\d+)?(?:/\d+)?%?)$", text)
    if is_match:
        return is_match.group(1)

    for p in [r"\d+\s+\d+/\d+", r"-?\d+/\d+", r"-?\d+(?:\.\d+)?%"]:
        m = re.search(p, text)
        if m:
            return m.group()

    m = re.search(r"\d+\s+\d+/\d+", text)
    if m:
        return m.group()

    m = re.search(r"-?\d+/\d+", text)
    if m:
        return m.group()

    m = re.search(r"-?\d+(?:\.\d+)?%", text)
    if m:
        return m.group()

    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m:
        return m.group()

    words = re.findall(r"[A-Za-z]+", text)
    if words:
        return words[-1]

    return text


def extract_final_answer(response):
    response = clean_text(response)
    boxed = extract_boxed(response)
    if boxed:
        return boxed

    pats = [
        r"final\s*answer\s*[:\-]?\s*(.*)",
        r"answer\s*[:\-]?\s*(.*)",
    ]
    for p in pats:
        m = re.findall(p, response, flags=re.I)
        if m:
            return extract_expression(m[-1].split("\n")[0])

    for line in reversed([x for x in response.splitlines() if x.strip()]):
        if line.lower().startswith(("step", "reasoning", "solution")):
            continue
        return extract_expression(line)

    return ""


def extract_reasoning(response):
    parts = re.split(r"final\s*answer\s*[:=\-]?", response, flags=re.I)
    return parts[0].strip()


def count_steps(response):
    s = re.findall(r"^\s*step\s+\d+\s*:", response, flags=re.I | re.M)
    if s:
        return len(s)
    s = re.findall(r"^\s*\d+[\.)]", response, flags=re.M)
    if s:
        return len(s)
    s = re.findall(r"^\s*[-*\u2022]", response, flags=re.M)
    if s:
        return len(s)
    return len([x for x in response.splitlines() if x.strip()])


def parse_response(response):
    return {
        "model_response": response,
        "reasoning": extract_reasoning(response),
        "model_final_answer": extract_final_answer(response),
        "model_step_count": count_steps(response),
    }

# MULTI-BRANCH ToT PARSING (new)
def parse_branch_response(branch_text):
    """Parse a single ToT candidate branch. Same shape as parse_response —
    each branch is itself a complete CoT-style attempt at the problem."""

    return parse_response(branch_text)


def parse_selection_response(selection_text):
    """Parse the model's branch-selection output.

    Expected format (see src/prompts/tot.py build_selection_prompt):
        Selected Branch: <int>
        Justification: <text>

    Falls back to branch 1 if the selection can't be parsed, so a malformed
    selection response degrades to "use the first candidate" rather than
    crashing the run.
    """

    text = clean_text(selection_text)

    branch_match = re.search(
        r"selected\s*branch\s*[:\-]?\s*(\d+)", text, flags=re.I
    )
    selected_branch = int(branch_match.group(1)) if branch_match else 1

    justification_match = re.search(
        r"justification\s*[:\-]?\s*(.*)", text, flags=re.I | re.S
    )
    justification = (
        justification_match.group(1).strip().split("\n")[0]
        if justification_match
        else ""
    )

    return {
        "selected_branch": selected_branch,
        "selection_justification": justification,
    }
