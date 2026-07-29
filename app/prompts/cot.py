def build_prompt(question):

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