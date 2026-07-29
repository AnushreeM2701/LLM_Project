import re

def clean_text(text):
    text = str(text)
    for t in ["```","**","__","###","##","#"]:
        text = text.replace(t,"")
    return text.strip()

def clean_answer(answer):
    answer = clean_text(answer)
    answer = answer.strip("$").replace(",", "")
    answer = answer.replace(r"\(","").replace(r"\)","")
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\boxed\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", answer)
    answer = re.sub(r"^[A-Za-z]+\s*=\s*", "", answer)
    answer = re.sub(r"\s+"," ",answer)
    return answer.strip(" .,")

def extract_boxed(response):
    # First try boxed fractions like \boxed{\frac{4}{3}}
    m = re.findall(
        r"\\boxed\{(\\frac\{[^{}]+\}\{[^{}]+\})\}",
        response
    )
    if m:
        return clean_answer(m[-1])

    # Then normal boxed values like \boxed{2}
    m = re.findall(
        r"\\boxed\{([^{}]+)\}",
        response
    )
    if m:
        return clean_answer(m[-1])

    return None

def extract_expression(text):
    text = clean_answer(text)
    for p in [r"\d+\s+\d+/\d+", r"-?\d+/\d+", r"-?\d+(?:\.\d+)?%"]:
        m = re.search(p,text)
        if m: return m.group()
    # Mixed numbers (e.g., 2 1/3)
    m = re.search(r"\d+\s+\d+/\d+", text)
    if m:
        return m.group()

    # Fractions (e.g., 4/3)
    m = re.search(r"-?\d+/\d+", text)
    if m:
        return m.group()

    # Percentages (e.g., 25%)
    m = re.search(r"-?\d+(?:\.\d+)?%", text)
    if m:
        return m.group()

    # Integers/decimals
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m:
        return m.group()
    words = re.findall(r"[A-Za-z]+", text)
    if words: return words[-1]
    return text

def extract_final_answer(response):
    response = clean_text(response)
    boxed = extract_boxed(response)
    if boxed:
        return boxed
    pats=[
        r"final\s*answer\s*[:\-]?\s*(.*)",
        r"answer\s*[:\-]?\s*(.*)"
    ]
    for p in pats:
        m=re.findall(p,response,flags=re.I)
        if m:
            return extract_expression(m[-1].split("\n")[0])
    for line in reversed([x for x in response.splitlines() if x.strip()]):
        if line.lower().startswith(("step","reasoning","solution")):
            continue
        return extract_expression(line)
    return ""

def extract_reasoning(response):
    parts = re.split(
    r"final\s*answer\s*[:=\-]?",
    response,
    flags=re.I
    )
    return parts[0].strip()

def count_steps(response):
    s=re.findall(r"^\s*step\s+\d+\s*:",response,flags=re.I|re.M)
    if s: return len(s)
    s=re.findall(r"^\s*\d+[\.)]",response,flags=re.M)
    if s: return len(s)
    s=re.findall(r"^\s*[-*•]",response,flags=re.M)
    if s: return len(s)
    return len([x for x in response.splitlines() if x.strip()])

def parse_response(response):
    return {
        "model_response": response,
        "reasoning": extract_reasoning(response),
        "model_final_answer": extract_final_answer(response),
        "model_step_count": count_steps(response)
    }
