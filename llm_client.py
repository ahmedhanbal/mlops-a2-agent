"""LLM clients for Anthropic and OpenAI-compatible APIs (including DeepSeek)."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from redaction import redact_text

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
REQUEST_TIMEOUT_S = 120.0
MAX_ATTEMPTS = 5


class LLMError(Exception):
    """The LLM could not be used (after retries)."""


class LLMConfigError(LLMError):
    """Missing or invalid LLM configuration (credentials, provider, model)."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


def load_env_file(path: Path) -> bool:
    """Minimal .env loader (KEY=VALUE, '#' comments). Never overrides real env vars."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def _classify(exc: Exception) -> str:
    """Return 'retry', 'auth' or 'fatal' for an SDK exception."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    if isinstance(status, int):
        if status in (401, 403):
            return "auth"
        if status in (408, 409, 425, 429) or status >= 500:
            return "retry"
        return "fatal"
    if "timeout" in name or "connection" in name or "ratelimit" in name or "overloaded" in name:
        return "retry"
    return "fatal"


def _retry_after(exc: Exception) -> Optional[float]:
    try:
        value = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(value) if value else None
    except Exception:
        return None


def _call_with_retries(fn, what: str):
    last: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except LLMError:
            raise
        except Exception as exc:
            kind = _classify(exc)
            msg = redact_text(f"{type(exc).__name__}: {exc}")[:400]
            if kind == "auth":
                raise LLMConfigError(f"{what}: authentication/permission error ({msg}). Check the API key and model access.")
            if kind == "fatal":
                raise LLMError(f"{what}: request rejected ({msg})")
            last = exc
            if attempt == MAX_ATTEMPTS:
                break
            delay = _retry_after(exc) or min(30.0, 2.0 ** attempt)
            time.sleep(min(delay, 60.0) + random.uniform(0, 0.5))
    raise LLMError(f"{what}: failed after {MAX_ATTEMPTS} attempts ({redact_text(f'{type(last).__name__}: {last}')[:300]})")


class AnthropicClient:
    def __init__(self, api_key: str, model: str, temperature: Optional[float], base_url: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMConfigError("The 'anthropic' package is not installed (pip install -r requirements.txt).") from exc
        kwargs = {"api_key": api_key, "timeout": REQUEST_TIMEOUT_S, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        self.model = model
        self.temperature = temperature
        self._send_temperature = temperature is not None

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int) -> LLMResponse:
        def once():
            params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": messages,
            }
            if self._send_temperature:
                params["temperature"] = self.temperature
            try:
                return self._client.messages.create(**params)
            except Exception as exc:
                if self._send_temperature and "temperature" in str(exc).lower() and getattr(exc, "status_code", 0) == 400:
                    self._send_temperature = False
                    params.pop("temperature", None)
                    return self._client.messages.create(**params)
                raise

        resp = _call_with_retries(once, "Anthropic API")
        text = "".join(getattr(block, "text", "") for block in resp.content if getattr(block, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        return LLMResponse(text, getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))


class OpenAIClient:
    def __init__(self, api_key: str, model: str, temperature: Optional[float], base_url: Optional[str] = None):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise LLMConfigError("The 'openai' package is not installed (pip install -r requirements.txt).") from exc
        kwargs = {"api_key": api_key, "timeout": REQUEST_TIMEOUT_S, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self._token_param = "max_tokens"
        self._send_temperature = temperature is not None

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int) -> LLMResponse:
        def once():
            for _ in range(3):
                params = {"model": self.model, "messages": [{"role": "system", "content": system}, *messages],
                          self._token_param: max_tokens}
                if self._send_temperature:
                    params["temperature"] = self.temperature
                try:
                    return self._client.chat.completions.create(**params)
                except Exception as exc:
                    text = str(exc).lower()
                    if getattr(exc, "status_code", 0) == 400:
                        if "max_completion_tokens" in text and self._token_param == "max_tokens":
                            self._token_param = "max_completion_tokens"
                            continue
                        if "temperature" in text and self._send_temperature:
                            self._send_temperature = False
                            continue
                    raise
            raise LLMError("could not find compatible request parameters for this model")

        resp = _call_with_retries(once, "OpenAI-compatible API")
        choice = resp.choices[0] if resp.choices else None
        text = (choice.message.content or "") if choice else ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(text, getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))


def build_llm(provider: Optional[str] = None, model: Optional[str] = None):
    """Create a client from CLI overrides and environment variables."""
    env = os.environ
    provider = (provider or env.get("LLM_PROVIDER") or "").strip().lower() or None
    keys = {
        "deepseek": env.get("DEEPSEEK_API_KEY", "").strip(),
        "anthropic": env.get("ANTHROPIC_API_KEY", "").strip(),
        "openai": env.get("OPENAI_API_KEY", "").strip(),
    }
    generic_key = env.get("LLM_API_KEY", "").strip()
    base_url = env.get("LLM_BASE_URL", "").strip() or None

    if provider in {"openai-compatible", "openai_compatible"}:
        provider = "openai"
    if provider is None:
        provider = next((name for name in ("deepseek", "anthropic", "openai") if keys[name]), "deepseek")
    if provider not in keys:
        raise LLMConfigError(f"Unsupported LLM_PROVIDER '{provider}'. Use 'deepseek', 'anthropic' or 'openai'.")

    key = keys[provider] or generic_key
    if not key:
        raise LLMConfigError(f"No API key for provider '{provider}'. Set {provider.upper()}_API_KEY (or LLM_API_KEY); see .env.example.")

    temperature: Optional[float] = None
    raw_temp = env.get("LLM_TEMPERATURE", "").strip()
    if raw_temp:
        try:
            temperature = float(raw_temp)
        except ValueError:
            raise LLMConfigError(f"LLM_TEMPERATURE must be a number, got '{raw_temp}'.")
    model = model or env.get("LLM_MODEL", "").strip() or None

    if provider == "anthropic":
        return AnthropicClient(key, model or DEFAULT_ANTHROPIC_MODEL, 0.0 if temperature is None else temperature, base_url)
    if provider == "deepseek":
        return OpenAIClient(key, model or DEFAULT_DEEPSEEK_MODEL, temperature, base_url or DEEPSEEK_BASE_URL)
    return OpenAIClient(key, model or DEFAULT_OPENAI_MODEL, temperature, base_url)
