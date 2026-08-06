"""
Shared return contract for every model client.

Every provider module (gemini.py, groq_gptoss.py, mistral_large.py) exposes
a generate_response(prompt) -> ModelResponse function with this exact shape,
so the experiment runner and ToT orchestration never need provider-specific
branching to read a result.
"""

from dataclasses import dataclass


@dataclass
class ModelResponse:
    text: str
    model_version: str   # exact served model version/string, for provenance
    latency_s: float
    raw_usage: str = ""  # provider-specific token usage info, stored as-is
