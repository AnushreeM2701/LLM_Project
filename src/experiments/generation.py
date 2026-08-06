"""
Shared retry/backoff/pacing wrapper around model generation.

Extracted so src/experiments/run_experiment.py (the real resumable
pipeline) and src/experiments/run_pilot.py (the pre-commit pilot) share
ONE retry implementation instead of two scripts silently drifting apart —
the run_pilot.py crash on a transient Gemini 503 (no retry logic at all)
is exactly the kind of duplication bug this rebuild is meant to prevent.

429 (quota) and 503 (server busy/unavailable) are handled differently on
exhaustion, matching the prior pipeline's reasoning: a 429 that survives
max_retries is very likely a genuine daily-quota wall, so it's worth
abandoning the rest of THIS model's questions for this run (the resumable
design picks them back up next run / after quota resets) rather than
burning 5 retries x wait_seconds on every remaining question. A 503 is
more often a transient blip on one request -- exhausting retries there
just means recording this one question as an empty (incorrect) response
and moving on, not abandoning the whole model.
"""

import time

from config.config import RETRY_SETTINGS, TOT_BRANCH_COUNT
from src.models.model_loader import get_model
from src.prompts import cot, tot


class QuotaExhausted(Exception):
    """429 survived max_retries -- caller should stop attempting further
    questions for this model in this run."""


_last_call_time = {}


def pace(provider: str) -> None:
    """Proactively sleep so we don't exceed a provider's free-tier rate
    limit -- reactive 429 backoff alone is far too slow for Mistral's 2
    requests/minute cap."""

    min_interval = RETRY_SETTINGS[provider]["min_request_interval_s"]
    last = _last_call_time.get(provider)

    if last is not None:
        elapsed = time.perf_counter() - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    _last_call_time[provider] = time.perf_counter()


def generate_with_retry(model_name: str, prompt_type: str, question: str) -> dict:
    """Runs one experiment's generation (1 call for CoT, N+1 for ToT) with
    retry/backoff. Raises QuotaExhausted if a 429 survives max_retries.
    Returns an empty-text dict if a 503 survives max_retries (record as a
    failed/incorrect response and move on)."""

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
            # Any 5xx is a transient server-side condition worth retrying --
            # narrowly matching "503" alone missed a Mistral 500 ("internal
            # server_error" / "Service unavailable") that killed a thread
            # mid-pilot. Word-boundary-ish check via surrounding non-digits
            # to avoid accidentally matching a 5xx substring inside an
            # unrelated large number.
            is_server_busy = any(
                f" {code}" in error or f"status {code}" in error_lower or f"code {code}" in error_lower
                for code in ("500", "502", "503", "504")
            )
            # Transient network issues (read timeouts, connection resets) --
            # discovered when a Mistral call raised a bare "read operation
            # timed out" mid-pilot, uncaught by the 429/503 checks alone,
            # which killed the whole thread instead of just retrying.
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
