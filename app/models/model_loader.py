from app.models.gemini import generate_response as gemini_response
from app.models.qwen import generate_response as qwen_response
from app.models.gpt_oss import generate_response as gpt_response


MODELS = {

    "gemini": gemini_response,

    "qwen": qwen_response,

    "gpt": gpt_response

}


def get_model(model_name):
    """
    Return the selected model.
    """

    model_name = model_name.lower()

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    return MODELS[model_name]