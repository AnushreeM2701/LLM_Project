def build_prompt(question):
    """
    Standard prompt without any explicit reasoning instructions.
    """

    prompt = f"""
Solve the following mathematics problem.
Provide your answer clearly.

Problem:

{question}
"""

    return prompt