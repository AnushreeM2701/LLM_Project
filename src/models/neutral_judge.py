"""
Client for the neutral error-classification judge (llama-3.3-70b-versatile,
via Groq -- originally Kimi K2 0905, but Groq removed it from their model
catalog entirely; see config.config.NEUTRAL_JUDGE_MODEL_ID).

Deliberately separate from src/models/groq_gptoss.py: that module is one of
the three STUDY models being evaluated; this one exists only to validate
the primary judge (Gemini, which IS one of the three study models — see
docs/limitations.md for why that overlap needed a validation check in the
first place). Using the same underlying provider (Groq) as the GPT-OSS
study model is fine — the thing that must stay independent is the specific
MODEL being judged vs. judging, not the hosting provider.
"""

import os
import time

from groq import Groq
from dotenv import load_dotenv

from config.config import NEUTRAL_JUDGE_MODEL_ID
from src.models.base import ModelResponse

load_dotenv()

# See src/models/gemini.py for why this timeout exists.
_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=120.0)


def generate_response(prompt: str, temperature: float = 0) -> ModelResponse:

    start = time.perf_counter()

    response = _client.chat.completions.create(
        model=NEUTRAL_JUDGE_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=1024,
    )

    latency_s = time.perf_counter() - start

    text = (response.choices[0].message.content or "").strip()

    model_version = getattr(response, "model", None) or NEUTRAL_JUDGE_MODEL_ID

    return ModelResponse(text=text, model_version=model_version, latency_s=latency_s)
