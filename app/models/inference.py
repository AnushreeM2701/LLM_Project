from app.prompts import cot, tot
from app.models.model_loader import get_model

# ==========================================================
# Prompt Builders
# ==========================================================

PROMPT_BUILDERS = {

    "cot": cot.build_prompt,

    "tot": tot.build_prompt

}

# ==========================================================
# Generate Response
# ==========================================================

def generate_response(model_name, prompt_type, question):
    """
    Generate a response using the selected
    model and prompting technique.

    Returns:
        str : Complete model response
    """

    model_name = model_name.lower()
    prompt_type = prompt_type.lower()

    # ------------------------------------------------------
    # Validate Prompt
    # ------------------------------------------------------

    if prompt_type not in PROMPT_BUILDERS:

        raise ValueError(
            f"Unknown prompt type: {prompt_type}"
        )

    # ------------------------------------------------------
    # Build Prompt
    # ------------------------------------------------------

    prompt = PROMPT_BUILDERS[prompt_type](question)

    # ------------------------------------------------------
    # Load Model
    # ------------------------------------------------------

    model = get_model(model_name)

    # ------------------------------------------------------
    # Generate Response
    # ------------------------------------------------------

    response = model(prompt)

    # ------------------------------------------------------
    # Ensure response is always a string
    # ------------------------------------------------------

    if response is None:

        return ""

    if not isinstance(response, str):

        response = str(response)

    return response.strip()


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    question = """
If a fair coin is tossed once, what is the probability of obtaining heads?
"""

    print(

        generate_response(

            model_name="gemini",

            prompt_type="cot",

            question=question

        )

    )