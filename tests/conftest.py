"""Every test runs offline: sockets are blocked, DEMO_MODE is on, no API keys,
and the LLM cache points at a temporary directory."""
import socket

import pytest

from app import llm

LLM_ENV = ["LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_BASE_URL", "GEMINI_MODEL",
           "GROQ_API_KEY", "GROQ_BASE_URL", "GROQ_MODEL"]


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(args)
        raise NetworkBlocked("network access attempted during tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setenv("DEMO_MODE", "true")
    for var in LLM_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
    yield attempts
    assert attempts == [], "a test tried to open a network connection"
