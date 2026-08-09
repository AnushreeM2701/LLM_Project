import time

from config.config import RETRY_SETTINGS, TOT_BRANCH_COUNT
from src.models.model_loader import get_model
from src.prompts import cot, tot


class QuotaExhausted(Exception):
    """429 survived max_retries -- caller should stop attempting further
    questions for this model in this run."""


_last_call_time = {}


def pace(provider: str) -> None:

    min_interval = RETRY_SETTINGS[provider]["min_request_interval_s"]
    last = _last_call_time.get(provider)

    if last is not None:
        elapsed = time.perf_counter() - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    _last_call_time[provider] = time.perf_counter()


def generate_with_retry(model_name: str, prompt_type: str, question: str) -> dict:

    settings = RETRY_SETTINGS[model_name]
    generate_fn = get_model(model_name)

    retry_count = 0

    while True:

        try:
            pace(model_name)

            if prompt_type == "cot":
                prompt = cot.build_prompt(question)
                result = generate_fn(prompt)
                return {
                    "final_text": result.text,
                    "model_version": result.model_version,
                    "latency_s": result.latency_s,
                    "branches": [],
                }

            elif prompt_type == "tot":
                tot_result = tot.generate_tot_response(question, generate_fn, TOT_BRANCH_COUNT)
                return {
                    "final_text": tot_result.final_response,
                    "model_version": tot_result.model_version,
                    "latency_s": tot_result.total_latency_s,
                    "branches": tot_result.branches,
                }

            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

        except Exception as e:

            error = str(e)
            error_lower = error.lower()
            is_quota = "429" in error
            is_server_busy = any(
                f" {code}" in error or f"status {code}" in error_lower or f"code {code}" in error_lower
                for code in ("500", "502", "503", "504")
            )
            is_network_issue = (
                "timed out" in error_lower
                or "timeout" in error_lower
                or "connection" in error_lower
            )

            if not (is_quota or is_server_busy or is_network_issue):
                raise

            retry_count += 1
            if is_quota:
                label = "Quota exceeded"
            elif is_server_busy:
                label = "Server busy/unavailable"
            else:
                label = "Network error"
            print(f"\n{label} ({retry_count}/{settings['max_retries']}): {error[:150]}")

            if retry_count >= settings["max_retries"]:

                if is_quota:
                    print(f"Daily quota likely exhausted for {model_name} -- "
                          f"skipping remaining {model_name} experiments this run.")
                    raise QuotaExhausted(model_name) from e

                print("Skipping this question after repeated server errors.")
                return {"final_text": "", "model_version": "", "latency_s": 0.0, "branches": []}

            time.sleep(settings["wait_seconds"])
