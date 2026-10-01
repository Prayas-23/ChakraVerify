"""Provider-agnostic LLM client, caching, retries, fallback (IMPLEMENTATION_PLAN.md §8).

The openai SDK is used as a generic client for OpenAI-compatible endpoints.
chat() tries the active provider (LLM_PROVIDER), retries 429/5xx with
exponential backoff, then falls back to the other provider. Successful
responses are cached in data/cache/llm/ keyed by SHA-256 of
(provider, model, messages, tools).

DEMO_MODE=true never touches the network: only cache hits are served, anything
else raises LLMUnavailable so callers fall back to ground truth / the
deterministic planner. API keys are never logged.
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "llm"
PROVIDERS = ("gemini", "groq")
DEFAULT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
}
MAX_ATTEMPTS = 3            # per provider
BACKOFF_SECONDS = 2.0       # 2 s, 4 s between attempts
REQUEST_TIMEOUT_S = 60
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_sleep = time.sleep  # replaced in tests


class LLMUnavailable(RuntimeError):
    """No provider could answer (demo-mode cache miss, no keys, or repeated failures)."""


def demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}


def provider_order() -> list[str]:
    primary = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if primary not in PROVIDERS:
        primary = "gemini"
    return [primary] + [p for p in PROVIDERS if p != primary]


def provider_config(name: str) -> dict:
    prefix = name.upper()
    return {
        "name": name,
        "api_key": os.getenv(f"{prefix}_API_KEY", "").strip(),
        "base_url": os.getenv(f"{prefix}_BASE_URL", "").strip() or DEFAULT_BASE_URLS[name],
        "model": os.getenv(f"{prefix}_MODEL", "").strip(),
    }


def cache_key(provider: str, model: str, messages: list[dict], tools: list[dict] | None) -> str:
    canonical = json.dumps([provider, model, messages, tools], sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_put(key: str, response: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(response, indent=2), encoding="utf-8")


def _make_client(cfg: dict):
    from openai import OpenAI
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"],
                  timeout=REQUEST_TIMEOUT_S, max_retries=0)


def _is_retryable(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) in RETRYABLE_STATUS:
        return True
    try:
        import openai
        return isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))
    except ImportError:
        return False


def _extra_content(tool_call) -> dict | None:
    """Provider-specific tool-call data, e.g. Gemini 3's extra_content.google.thought_signature,
    which must be sent back unchanged on the next turn (missing → HTTP 400)."""
    extra = getattr(tool_call, "extra_content", None)
    if extra is None:
        extra = (getattr(tool_call, "model_extra", None) or {}).get("extra_content")
    return extra if isinstance(extra, dict) else None


def _normalise(message) -> dict:
    tool_calls = []
    for i, tc in enumerate(getattr(message, "tool_calls", None) or []):
        call = {"id": tc.id or f"call_{i}", "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
        extra = _extra_content(tc)
        if extra is not None:
            call["extra_content"] = extra
        tool_calls.append(call)
    return {"content": message.content, "tool_calls": tool_calls}


def _call(cfg: dict, messages: list[dict], tools: list[dict] | None, response_format: dict | None) -> dict:
    kwargs = {"model": cfg["model"], "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_format:
        kwargs["response_format"] = response_format
    response = _make_client(cfg).chat.completions.create(**kwargs)
    return _normalise(response.choices[0].message)


def chat(messages: list[dict], tools: list[dict] | None = None,
         response_format: dict | None = None) -> dict:
    """Return {"content", "tool_calls": [{"id", "name", "arguments"}], "provider", "model", "cached"}."""
    errors = []
    for name in provider_order():
        cfg = provider_config(name)
        if not cfg["model"]:
            errors.append(f"{name}: no model configured")
            continue
        key = cache_key(name, cfg["model"], messages, tools)
        cached = _cache_get(key)
        if cached is not None:
            return {**cached, "provider": name, "model": cfg["model"], "cached": True}
        if demo_mode():
            errors.append(f"{name}: no cached response (DEMO_MODE)")
            continue
        if not cfg["api_key"]:
            errors.append(f"{name}: no API key")
            continue

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = _call(cfg, messages, tools, response_format)
            except Exception as exc:  # provider SDK errors vary; classify, never log the key
                status = getattr(exc, "status_code", None)
                retry = _is_retryable(exc) and attempt < MAX_ATTEMPTS
                log.warning("LLM %s/%s attempt %d failed: %s (status %s)%s", name, cfg["model"],
                            attempt, type(exc).__name__, status, "; retrying" if retry else "")
                if retry:
                    _sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
                    continue
                errors.append(f"{name}: {type(exc).__name__} (status {status})")
                break
            _cache_put(key, result)
            return {**result, "provider": name, "model": cfg["model"], "cached": False}

    raise LLMUnavailable("; ".join(errors) or "no LLM provider configured")
