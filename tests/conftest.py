"""Every test runs offline: non-loopback sockets are blocked, DEMO_MODE is on,
no API keys, and the LLM cache points at a temporary directory.

Loopback is allowed because the asyncio event loop behind FastAPI's TestClient
opens a local socket pair on Windows; nothing can reach the outside network."""
import socket

import pytest

from app import llm

LLM_ENV = ["LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_BASE_URL", "GEMINI_MODEL",
           "GROQ_API_KEY", "GROQ_BASE_URL", "GROQ_MODEL"]
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    attempts = []
    real_connect = socket.socket.connect

    def guarded_connect(sock, address, *args, **kwargs):
        if isinstance(address, tuple) and address and address[0] in LOOPBACK:
            return real_connect(sock, address, *args, **kwargs)
        attempts.append(address)
        raise NetworkBlocked(f"network access attempted during tests: {address!r}")

    def blocked(address, *args, **kwargs):
        attempts.append(address)
        raise NetworkBlocked(f"network access attempted during tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setenv("DEMO_MODE", "true")
    for var in LLM_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
    yield attempts
    assert attempts == [], "a test tried to open a network connection"
