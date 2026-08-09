from dataclasses import dataclass


@dataclass
class ModelResponse:
    text: str
    model_version: str 
    latency_s: float
    raw_usage: str = ""  # provider-specific token usage info, stored as-is
