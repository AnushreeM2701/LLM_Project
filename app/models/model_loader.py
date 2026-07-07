from app.models.gemini import generate_response as gemini_response

# Uncomment later
# from app.models.qwen import generate_response as qwen_response
# from app.models.deepseek import generate_response as deepseek_response


def get_model(model_name):
    """
    Return the appropriate model function.
    """

    model_name = model_name.lower()

    if model_name == "gemini":
        return gemini_response

    # elif model_name == "qwen":
    #     return qwen_response

    # elif model_name == "deepseek":
    #     return deepseek_response

    else:
        raise ValueError(f"Unsupported model: {model_name}")