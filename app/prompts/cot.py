def build_prompt(question):
    """
    Chain-of-Thought Prompt
    """

    prompt = f"""
Solve the following mathematics problem.

Think through the solution step by step before giving your final answer.

At the end, clearly state the final answer.

Problem:

{question}
"""

    return prompt