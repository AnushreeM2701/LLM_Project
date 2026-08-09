import os
import time

from mistralai.client import Mistral
from dotenv import load_dotenv

from config.config import MODELS
from src.models.base import ModelResponse

load_dotenv()

_CFG = MODELS["mistral"]

# Request timeout (ms) -- see src/models/gemini.py for why this exists.
_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"), timeout_ms=120_000)


def generate_response(prompt: str, temperature: float = None, max_tokens: int = None) -> ModelResponse:

    if temperature is None:
        temperature = _CFG["temperature"]

    if max_tokens is None:
        max_tokens = _CFG["max_tokens"]

    start = time.perf_counter()

    response = _client.chat.complete(
        model=_CFG["model_id"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    latency_s = time.perf_counter() - start

    text = (response.choices[0].message.content or "").strip()

    model_version = getattr(response, "model", None) or _CFG["model_id"]

    usage = getattr(response, "usage", None)

    return ModelResponse(
        text=text,
        model_version=model_version,
        latency_s=latency_s,
        raw_usage=str(usage) if usage else "",
    )


if __name__ == "__main__":

    question = "If a fair coin is tossed once, what is the probability of obtaining heads?"

    result = generate_response(question)
    print(result.text)
    print("model_version:", result.model_version)
    print("latency_s:", result.latency_s)
