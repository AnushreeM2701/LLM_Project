"""
Groq client for GPT-OSS-120B. Unified return contract — see src/models/base.py.

Replaces the prior pipeline's llama-3.3-70b-versatile. Llama 4 Maverick was
the originally planned replacement, but was found (mid-build, while writing
this module) to have been deprecated by Groq in Feb 2026, with its successor
Llama 4 Scout also deprecated in Jun 2026. GPT-OSS-120B is Groq's own
recommended migration target for both.

reasoning_effort is pinned to "low" and held constant across CoT and ToT
(see config.config.MODELS) — GPT-OSS is reasoning-native, so this doesn't
eliminate internal reasoning, but it does hold it fixed so the CoT/ToT
prompting comparison isn't confounded by a variable reasoning budget. See
docs/limitations.md.
"""

import os
import time

from groq import Groq
from dotenv import load_dotenv

from config.config import MODELS
from src.models.base import ModelResponse

load_dotenv()

_CFG = MODELS["groq"]

# Request timeout (s) -- see src/models/gemini.py for why this exists: a
# hung network call raises no exception on its own, so without a timeout
# generate_with_retry() never gets a chance to retry/skip.
_REQUEST_TIMEOUT_S = 120.0

_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=_REQUEST_TIMEOUT_S)


def generate_response(prompt: str, temperature: float = None, max_tokens: int = None) -> ModelResponse:

    if temperature is None:
        temperature = _CFG["temperature"]

    if max_tokens is None:
        max_tokens = _CFG["max_tokens"]

    start = time.perf_counter()

    response = _client.chat.completions.create(
        model=_CFG["model_id"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        reasoning_effort=_CFG["reasoning_effort"],
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
