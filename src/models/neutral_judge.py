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
