def build_prompt(question):
    """
    Tree-of-Thought Prompt
    """

    prompt = f"""
Solve the following mathematics problem.

Before deciding on the final answer:

1. Consider multiple possible solution strategies.
2. Evaluate which strategy is most appropriate.
3. Follow the best strategy step by step.
4. If necessary, revise your reasoning before producing the final answer.

Finally, provide the final answer clearly.

Problem:

{question}
"""

    return prompt