from src.models.gemini import generate_response as gemini_response
from src.models.groq_gptoss import generate_response as groq_response
from src.models.mistral_large import generate_response as mistral_response

MODELS = {
    "gemini": gemini_response,
    "groq": groq_response,
    "mistral": mistral_response,
}


def get_model(model_name: str):
    """Return the generate_response(prompt) -> ModelResponse function for
    the given model key ("gemini", "groq", "mistral")."""

    model_name = model_name.lower()

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    return MODELS[model_name]
