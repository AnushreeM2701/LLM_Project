"""
Gemini client. Unified return contract shared by every model module:
{text, model_version, latency_s, raw_usage} — see src/models/base.py.

Uses gemini-3-flash (upgraded from the prior pipeline's gemini-3.1-flash-lite)
with thinking_level pinned to "minimal" and held constant across CoT and ToT
so the prompting-strategy manipulation, not a variable internal reasoning
budget, is what's being measured (see config.config.MODELS and
docs/limitations.md).
"""

import os
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

from config.config import MODELS
from src.models.base import ModelResponse

load_dotenv()

_CFG = MODELS["gemini"]

# Request timeout (ms) -- without this, a hung network call can block a
# thread indefinitely with no exception raised, so generate_with_retry()
# never gets a chance to retry. Discovered when a Gemini call stalled for
# ~58 minutes with no error during the Medium-tier run. 120s is generous
# relative to observed real call durations (even a heavy ToT branch call
# has taken well under 2 minutes) while still bounding the worst case.
_REQUEST_TIMEOUT_MS = 120_000

_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
)


def generate_response(prompt: str, temperature: float = None, max_tokens: int = None) -> ModelResponse:

    if temperature is None:
        temperature = _CFG["temperature"]

    if max_tokens is None:
        max_tokens = _CFG["max_tokens"]

    start = time.perf_counter()

    response = _client.models.generate_content(
        model=_CFG["model_id"],
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=_CFG["thinking_level"],
            ),
        ),
    )

    latency_s = time.perf_counter() - start

    text = response.text.strip() if response.text else ""

    model_version = getattr(response, "model_version", None) or _CFG["model_id"]

    usage = getattr(response, "usage_metadata", None)

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
