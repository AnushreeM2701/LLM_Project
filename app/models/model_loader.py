from app.models.gemini import generate_response as gemini_response
from app.models.groq_model import generate_response as groq_response
from app.models.mistral_model import generate_response as mistral_response


MODELS = {

    "gemini": gemini_response,

    "groq": groq_response,

    "mistral": mistral_response

}


def get_model(model_name):
    """
    Return the selected model.
    """

    model_name = model_name.lower()

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    return MODELS[model_name]