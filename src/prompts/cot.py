"""
Single-call Chain-of-Thought prompt. Ported from the prior pipeline's
app/prompts/cot.py, unchanged — this was already a sound, standard CoT
prompt design.
"""


def build_prompt(question: str) -> str:

    return f"""
Solve the following mathematics problem.

Show your reasoning as clear, numbered steps.

Format your response exactly as:

Step 1: ...
Step 2: ...
Step 3: ...

Final Answer: <answer>

Problem:
{question}
"""
