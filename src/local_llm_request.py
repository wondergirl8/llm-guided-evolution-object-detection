"""Bounded HTTP requests to the local LLM server."""

import os
import time

import requests


def _positive_number(name, default, cast):
    value = cast(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def post_with_retry(url, *, headers, payload, sleep=time.sleep):
    """Retry one transient transport failure; never retry a completed response."""
    connect = _positive_number("LLMGE_LLM_CONNECT_TIMEOUT_SECONDS", "10", float)
    read = _positive_number("LLMGE_LLM_READ_TIMEOUT_SECONDS", "1800", float)
    attempts = _positive_number("LLMGE_LLM_REQUEST_ATTEMPTS", "2", int)
    for attempt in range(1, attempts + 1):
        try:
            return requests.post(url, headers=headers, json=payload,
                                 timeout=(connect, read))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt == attempts:
                raise RuntimeError(
                    f"Local LLM request failed after {attempts} attempt(s) "
                    f"(connect timeout {connect}s, read timeout {read}s)"
                ) from exc
            print(f"Local LLM request attempt {attempt}/{attempts} failed: {exc}; retrying", flush=True)
            sleep(2)
