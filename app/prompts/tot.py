def build_prompt(question):

    return f"""
Solve the following mathematics problem.

Consider more than one possible approach before selecting the best one.

Show only the reasoning for the final selected approach as clear, numbered steps.

Format your response exactly as:

Step 1: ...
Step 2: ...
Step 3: ...

Final Answer: <answer>

Problem:
{question}
"""