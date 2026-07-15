from app.prompts import baseline, cot, tot
from app.models.model_loader import get_model


def generate_response(model_name, prompt_type, question):
    """
    Generate a response using the selected
    model and prompting technique.
    """

    prompt_type = prompt_type.lower()

    # ----------------------------
    # Build Prompt
    # ----------------------------

    PROMPT_BUILDERS = {
    "cot": cot.build_prompt,
    "tot": tot.build_prompt,
    }

    prompt = PROMPT_BUILDERS[prompt_type](question)

    # ----------------------------
    # Load Model
    # ----------------------------

    model = get_model(model_name)

    # ----------------------------
    # Generate Response
    # ----------------------------

    response = model(prompt)

    return response


if __name__ == "__main__":

    question = """
If a fair coin is tossed once,
what is the probability of getting Heads?
"""

    response = generate_response(
        model_name="gemini",
        prompt_type="baseline",
        question=question
    )

    print(response)