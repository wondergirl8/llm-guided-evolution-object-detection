"""A slow or disconnected LLM server cannot block the operator forever."""

from types import SimpleNamespace

import pytest
import requests

from src import local_llm_request


def test_success_has_bounded_timeouts_and_no_retry(monkeypatch):
    calls = []
    response = SimpleNamespace(status_code=200)
    monkeypatch.setattr(local_llm_request.requests, "post",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or response)
    assert local_llm_request.post_with_retry("http://server/generate",
                                             headers={}, payload={}) is response
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == (10.0, 1800.0)


def test_timeout_retries_once_then_succeeds(monkeypatch):
    calls = []
    response = SimpleNamespace(status_code=200)

    def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ReadTimeout("stalled")
        return response

    monkeypatch.setattr(local_llm_request.requests, "post", post)
    delays = []
    assert local_llm_request.post_with_retry("http://server/generate", headers={},
                                             payload={}, sleep=delays.append) is response
    assert len(calls) == 2
    assert delays == [2]


def test_second_timeout_fails_the_operator(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        raise requests.exceptions.ReadTimeout("stalled")

    monkeypatch.setattr(local_llm_request.requests, "post", post)
    with pytest.raises(RuntimeError, match="after 2 attempt"):
        local_llm_request.post_with_retry("http://server/generate", headers={},
                                          payload={}, sleep=lambda _: None)
    assert len(calls) == 2


def test_completed_http_response_is_not_retried(monkeypatch):
    calls = []
    response = SimpleNamespace(status_code=500)
    monkeypatch.setattr(local_llm_request.requests, "post",
                        lambda *args, **kwargs: calls.append(1) or response)
    assert local_llm_request.post_with_retry("http://server/generate",
                                             headers={}, payload={}) is response
    assert len(calls) == 1
