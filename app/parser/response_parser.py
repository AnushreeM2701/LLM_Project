import re


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    text = str(text)

    text = text.replace("```", "")
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("###", "")
    text = text.replace("##", "")
    text = text.replace("#", "")

    return text.strip()


# ==========================================================
# CLEAN ANSWER
# ==========================================================

def clean_answer(answer):

    answer = clean_text(answer)

    if not answer:
        return ""

    # remove $
    answer = answer.strip("$")

    # remove punctuation
    answer = answer.rstrip(".")
    answer = answer.rstrip(",")

    # remove escaped brackets
    answer = answer.replace(r"\(", "")
    answer = answer.replace(r"\)", "")

    # latex spacing
    answer = answer.replace(r"\,", " ")
    answer = answer.replace(r"\;", " ")
    answer = answer.replace(r"\!", "")

    # ------------------------------------------
    # \text{Gray}
    # ------------------------------------------

    answer = re.sub(

        r"\\text\{([^}]*)\}",

        r"\1",

        answer

    )

    # ------------------------------------------
    # \boxed{}
    # ------------------------------------------

    answer = re.sub(

        r"\\boxed\{([^}]*)\}",

        r"\1",

        answer

    )

    # ------------------------------------------
    # \frac{31}{90}
    # ------------------------------------------

    answer = re.sub(

        r"\\frac\{([^}]*)\}\{([^}]*)\}",

        r"\1/\2",

        answer

    )

    # ------------------------------------------
    # variable assignment
    # x = 5
    # m = 2
    # ------------------------------------------

    answer = re.sub(

        r"^[A-Za-z]+\s*=\s*",

        "",

        answer

    )

    # ------------------------------------------
    # remove units
    # ------------------------------------------

    answer = re.sub(

        r"\b(minutes?|minute|hours?|hour|seconds?|km|km/h|meters?)\b",

        "",

        answer,

        flags=re.IGNORECASE

    )

    answer = re.sub(

        r"\s+",

        " ",

        answer

    )

    return answer.strip()


# ==========================================================
# EXTRACT BOXED
# ==========================================================

def extract_boxed(response):

    start = response.rfind(r"\boxed{")

    if start == -1:

        return None

    i = start + len(r"\boxed{")

    depth = 1

    answer = ""

    while i < len(response):

        c = response[i]

        if c == "{":

            depth += 1

        elif c == "}":

            depth -= 1

            if depth == 0:

                break

        answer += c

        i += 1

    return clean_answer(answer)


# ==========================================================
# EXTRACT MATHEMATICAL EXPRESSION
# ==========================================================

def extract_expression(text):

    text = clean_answer(text)

    # mixed fraction

    mixed = re.search(

        r"\d+\s+\d+/\d+",

        text

    )

    if mixed:

        return mixed.group()

    # fraction

    frac = re.search(

        r"\d+/\d+",

        text

    )

    if frac:

        return frac.group()

        # ------------------------------------------
    # Mixed fraction
    # ------------------------------------------

    mixed = re.search(
        r"\d+\s+\d+/\d+",
        text
    )

    if mixed:
        return mixed.group()

    # ------------------------------------------
    # Fraction
    # ------------------------------------------

    frac = re.search(
        r"\d+/\d+",
        text
    )

    if frac:
        return frac.group()

    # ------------------------------------------
    # Polynomial / equation
    # Example:
    # 24x^2-6x+3
    # ------------------------------------------

    poly = re.search(
        r"[0-9A-Za-z\^\+\-\*/]+(?:=[0-9A-Za-z\^\+\-\*/]+)?",
        text
    )

    if poly and any(op in poly.group() for op in ["+", "-", "^", "*", "/"]):
        return clean_answer(poly.group())

    # ------------------------------------------
    # Standalone number
    # Take LAST number
    # ------------------------------------------

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text
    )

    if numbers:
        return numbers[-1]

    # ------------------------------------------
    # Standalone word
    # Take LAST word
    # Gray
    # Saturday
    # ------------------------------------------

    words = re.findall(
        r"[A-Za-z]+",
        text
    )

    if words:
        return words[-1]

    return clean_answer(text)

# ==========================================================
# EXTRACT FINAL ANSWER
# ==========================================================

def extract_final_answer(response):

    response = clean_text(response)

    if not response:
        return ""

    # ------------------------------------------------------
    # 1. BOXED ANSWER
    # ------------------------------------------------------

    boxed = extract_boxed(response)

    if boxed:
        return boxed

    # ------------------------------------------------------
    # 2. COMMON FINAL ANSWER PATTERNS
    # ------------------------------------------------------

    patterns = [

        r"final\s*answer\s*[:\-]?\s*(.*)",

        r"final\s*answer\s*is\s*(.*)",

        r"the\s*final\s*answer\s*is\s*(.*)",

        r"answer\s*[:\-]?\s*(.*)",

        r"answer\s*is\s*(.*)",

        r"therefore\s*,?\s*the\s*answer\s*is\s*(.*)",

        r"thus\s*,?\s*the\s*answer\s*is\s*(.*)"

    ]

    for pattern in patterns:

        matches = re.findall(

            pattern,

            response,

            flags=re.IGNORECASE

        )

        if matches:

            candidate = matches[-1]

            candidate = candidate.split("\n")[0]

            candidate = clean_answer(candidate)

            # remove leading phrases

            candidate = re.sub(

                r"^(the\s+)?(final\s+)?answer\s+is\s+",

                "",

                candidate,

                flags=re.IGNORECASE

            )

            candidate = re.sub(

                r"^(the\s+)?answer\s+",

                "",

                candidate,

                flags=re.IGNORECASE

            )

            candidate = re.sub(

                r"^is\s+",

                "",

                candidate,

                flags=re.IGNORECASE

            )

            candidate = extract_expression(candidate)

            return candidate

    # ------------------------------------------------------
    # 3. SEARCH FROM BOTTOM
    # ------------------------------------------------------

    lines = [

        clean_answer(line)

        for line in response.splitlines()

        if line.strip()

    ]

    ignore = [

        "step",

        "solution",

        "reasoning",

        "calculation",

        "therefore"

    ]

    for line in reversed(lines):

        if len(line) > 200:
            continue

        if any(

            line.lower().startswith(word)

            for word in ignore

        ):

            continue

        candidate = extract_expression(line)

        if candidate:

            return candidate

    return ""


# ==========================================================
# COUNT STEPS
# ==========================================================

def count_steps(response):

    response = str(response)

    numbered = re.findall(

        r'^\s*\d+[\.\)]',

        response,

        flags=re.MULTILINE

    )

    if numbered:

        return len(numbered)

    bullet_points = re.findall(

        r'^\s*[-•*]',

        response,

        flags=re.MULTILINE

    )

    if bullet_points:

        return len(bullet_points)

    lines = [

        line.strip()

        for line in response.splitlines()

        if line.strip()

    ]

    return len(lines)


# ==========================================================
# PARSE RESPONSE
# ==========================================================

def parse_response(response):

    final_answer = extract_final_answer(response)

    return {

        "model_response": response,

        "model_final_answer": final_answer,

        "model_step_count": count_steps(response)

    }