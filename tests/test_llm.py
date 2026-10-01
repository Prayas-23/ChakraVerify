import logging
from types import SimpleNamespace

import pytest

from app import llm

MESSAGES = [{"role": "user", "content": "hello"}]
TOOLS = [{"type": "function", "function": {"name": "finish", "parameters": {"type": "object"}}}]


class StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def response(content="hi", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Plays back a list of responses / exceptions and records calls."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def live(monkeypatch, clients: dict):
    """Switch to live mode with both providers configured and fake clients."""
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret-key")
    monkeypatch.setenv("GROQ_MODEL", "groq-test-model")
    monkeypatch.setattr(llm, "_make_client", lambda cfg: clients[cfg["name"]])


def test_demo_mode_never_creates_a_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")

    def boom(cfg):
        raise AssertionError("client created in DEMO_MODE")

    monkeypatch.setattr(llm, "_make_client", boom)
    with pytest.raises(llm.LLMUnavailable, match="DEMO_MODE"):
        llm.chat(MESSAGES)


def test_demo_mode_without_any_keys_raises_cleanly():
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(MESSAGES, tools=TOOLS)


def test_demo_mode_serves_cache_hit(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    key = llm.cache_key("gemini", "gemini-test-model", MESSAGES, None)
    llm._cache_put(key, {"content": "cached answer", "tool_calls": []})
    result = llm.chat(MESSAGES)
    assert result["content"] == "cached answer"
    assert result["cached"] is True and result["provider"] == "gemini"


def test_cache_key_is_deterministic():
    a = llm.cache_key("gemini", "m", [{"role": "user", "content": "x"}], TOOLS)
    b = llm.cache_key("gemini", "m", [{"content": "x", "role": "user"}], TOOLS)
    assert a == b and len(a) == 64
    assert a != llm.cache_key("groq", "m", [{"role": "user", "content": "x"}], TOOLS)
    assert a != llm.cache_key("gemini", "other", [{"role": "user", "content": "x"}], TOOLS)
    assert a != llm.cache_key("gemini", "m", [{"role": "user", "content": "x"}], None)


def test_live_call_is_cached_and_reused(monkeypatch):
    gemini = FakeClient([response("first")])
    live(monkeypatch, {"gemini": gemini, "groq": FakeClient([])})
    first = llm.chat(MESSAGES)
    second = llm.chat(MESSAGES)
    assert (first["content"], first["cached"]) == ("first", False)
    assert (second["content"], second["cached"]) == ("first", True)
    assert len(gemini.calls) == 1
    assert gemini.calls[0]["model"] == "gemini-test-model"


def test_tool_calls_are_normalised(monkeypatch):
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="finish", arguments='{"summary": "x"}'))
    live(monkeypatch, {"gemini": FakeClient([response(None, [call])]), "groq": FakeClient([])})
    result = llm.chat(MESSAGES, tools=TOOLS)
    assert result["tool_calls"] == [{"id": "c1", "name": "finish", "arguments": '{"summary": "x"}'}]


def test_retries_429_then_falls_back_to_groq(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    gemini = FakeClient([StatusError(429)] * llm.MAX_ATTEMPTS)
    groq = FakeClient([response("from groq")])
    live(monkeypatch, {"gemini": gemini, "groq": groq})
    result = llm.chat(MESSAGES)
    assert result["provider"] == "groq" and result["content"] == "from groq"
    assert len(gemini.calls) == llm.MAX_ATTEMPTS
    assert sleeps == [2.0, 4.0]


def test_5xx_recovers_on_retry(monkeypatch):
    gemini = FakeClient([StatusError(503), response("ok")])
    live(monkeypatch, {"gemini": gemini, "groq": FakeClient([])})
    assert llm.chat(MESSAGES)["provider"] == "gemini"
    assert len(gemini.calls) == 2


def test_non_retryable_error_falls_back_without_retry(monkeypatch):
    gemini = FakeClient([StatusError(400)])
    live(monkeypatch, {"gemini": gemini, "groq": FakeClient([response("groq")])})
    assert llm.chat(MESSAGES)["provider"] == "groq"
    assert len(gemini.calls) == 1


def test_all_providers_failing_raises(monkeypatch):
    live(monkeypatch, {"gemini": FakeClient([StatusError(500)] * 3), "groq": FakeClient([StatusError(500)] * 3)})
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(MESSAGES)


def test_llm_provider_env_selects_primary(monkeypatch):
    live(monkeypatch, {"gemini": FakeClient([]), "groq": FakeClient([response("groq first")])})
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert llm.chat(MESSAGES)["provider"] == "groq"


def test_api_keys_never_logged(monkeypatch, caplog):
    live(monkeypatch, {"gemini": FakeClient([StatusError(500)] * 3), "groq": FakeClient([StatusError(429)] * 3)})
    with caplog.at_level(logging.DEBUG), pytest.raises(llm.LLMUnavailable) as exc_info:
        llm.chat(MESSAGES)
    assert caplog.records
    for secret in ("gemini-secret-key", "groq-secret-key"):
        assert secret not in caplog.text
        assert secret not in str(exc_info.value)
