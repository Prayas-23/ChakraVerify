"""Every test runs offline: non-loopback sockets are blocked, DEMO_MODE is on,
no API keys, and the LLM cache points at a temporary directory.

Loopback is allowed because the asyncio event loop behind FastAPI's TestClient
opens a local socket pair on Windows; nothing can reach the outside network.

Session-wide safety (covers module-scoped fixtures, which are created before the
per-test `offline` fixture): the environment below is set when pytest imports this
file — before `app.llm` runs load_dotenv(), which never overrides an existing
variable, so a developer's .env (real keys, DEMO_MODE=false) is ignored for the
whole test session. pytest_configure adds a session-wide network guard."""
import os
import socket

LLM_ENV = ["LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_BASE_URL", "GEMINI_MODEL",
           "GROQ_API_KEY", "GROQ_BASE_URL", "GROQ_MODEL"]

os.environ["DEMO_MODE"] = "true"
for _var in LLM_ENV:
    os.environ[_var] = ""   # empty = unset for the app; present = load_dotenv() leaves it alone

import pytest  # noqa: E402

from app import llm  # noqa: E402  (imported after the environment is forced)

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class NetworkBlocked(RuntimeError):
    pass


_session_saved: dict = {}


def pytest_configure(config):
    """Block non-loopback connections for the entire session, before any fixture runs."""
    real_connect, real_create = socket.socket.connect, socket.create_connection

    def session_connect(sock, address, *args, **kwargs):
        if isinstance(address, tuple) and address and address[0] in LOOPBACK:
            return real_connect(sock, address, *args, **kwargs)
        raise NetworkBlocked(f"network access attempted during tests: {address!r}")

    def session_create(address, *args, **kwargs):
        if isinstance(address, tuple) and address and address[0] in LOOPBACK:
            return real_create(address, *args, **kwargs)
        raise NetworkBlocked(f"network access attempted during tests: {address!r}")

    _session_saved.update(connect=real_connect, create_connection=real_create)
    socket.socket.connect = session_connect
    socket.create_connection = session_create


def pytest_unconfigure(config):
    if _session_saved:
        socket.socket.connect = _session_saved["connect"]
        socket.create_connection = _session_saved["create_connection"]


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
