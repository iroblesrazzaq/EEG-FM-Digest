from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Protocol

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_PROVIDER_ALIASES = {"google", "google_ai_studio", "gemini"}
OPENAI_COMPAT_PROVIDER_ALIASES = {"openai", "openrouter"} | GOOGLE_PROVIDER_ALIASES
DEFAULT_RATE_LIMIT_RETRIES = 5
DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True)
class LLMCallConfig:
    provider: str
    api_key: str
    model: str
    temperature: float
    max_output_tokens: int
    base_url: str | None = None


@dataclass(frozen=True)
class LLMCallResult:
    text: str
    provider: str
    model: str
    raw: Any | None = None


class LLMCaller(Protocol):
    def call(self, prompt: str, schema: dict[str, Any] | None = None) -> LLMCallResult: ...

    def count_tokens(self, content: str) -> int: ...

    def close(self) -> None: ...


class LLMRateLimitError(RuntimeError):
    pass


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    if normalized in GOOGLE_PROVIDER_ALIASES:
        return "google"
    return normalized


def infer_provider_from_env(default: str = "openrouter") -> str:
    explicit = os.environ.get("LLM_PROVIDER") or os.environ.get("LLM_API_PROVIDER")
    if explicit:
        return normalize_provider(explicit)
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "google"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return normalize_provider(default)


def provider_base_url(provider: str) -> str | None:
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return OPENROUTER_BASE_URL
    if normalized == "google":
        return GOOGLE_OPENAI_BASE_URL
    return None


def provider_supports_json_object(provider: str) -> bool:
    return normalize_provider(provider) in {"openai", "openrouter"}


class OpenAICall:
    """OpenAI-compatible chat-completions wrapper.

    This is used with OpenRouter and Google AI Studio by setting `base_url`.
    """

    def __init__(self, config: LLMCallConfig):
        normalized_provider = normalize_provider(config.provider)
        self.config = replace(
            config,
            provider=normalized_provider,
            base_url=config.base_url or provider_base_url(normalized_provider),
        )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing `openai` package. Install project dependencies first.") from exc

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()

    def count_tokens(self, content: str) -> int:
        # Approximate token count for routing fulltext vs slices.
        return max(1, len(content) // 4)

    def _extract_text_from_part(self, part: Any) -> str:
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                return text
            if isinstance(text, dict):
                nested = text.get("value")
                if isinstance(nested, str):
                    return nested
            if isinstance(part.get("content"), str):
                return str(part["content"])
            return ""
        text = getattr(part, "text", None)
        if isinstance(text, str):
            return text
        if isinstance(text, dict):
            nested = text.get("value")
            if isinstance(nested, str):
                return nested
        content = getattr(part, "content", None)
        if isinstance(content, str):
            return content
        return ""

    def _extract_text_from_dump(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [self._extract_text_from_part(item) for item in content]
            return "".join(part for part in parts if part).strip()
        return ""

    def _extract_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            model_dump = getattr(response, "model_dump", None)
            if callable(model_dump):
                return self._extract_text_from_dump(model_dump())
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = self._extract_text_from_part(item)
                if text:
                    parts.append(str(text))
            return "".join(parts).strip()
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            return self._extract_text_from_dump(model_dump())
        return ""

    def call(self, prompt: str, schema: dict[str, Any] | None = None) -> LLMCallResult:
        req: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if schema is not None and provider_supports_json_object(self.config.provider):
            req["response_format"] = {"type": "json_object"}

        backoff_seconds = float(os.environ.get("LLM_RATE_LIMIT_BACKOFF_SECONDS", str(DEFAULT_RATE_LIMIT_BACKOFF_SECONDS)))
        retry_count = int(os.environ.get("LLM_RATE_LIMIT_RETRIES", str(DEFAULT_RATE_LIMIT_RETRIES)))
        for attempt in range(retry_count + 1):
            try:
                response = self._client.chat.completions.create(**req)
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                response = getattr(exc, "response", None)
                if status_code is None and response is not None:
                    status_code = getattr(response, "status_code", None)
                if status_code in {402, 429}:
                    detail = ""
                    if response is not None:
                        detail = str(getattr(response, "text", "") or "")[:220]
                    if attempt < retry_count:
                        time.sleep(backoff_seconds * (2**attempt))
                        continue
                    raise LLMRateLimitError(
                        f"openai_compat_rate_limit_or_quota status={status_code} body={detail}"
                    ) from exc
                raise

        text = self._extract_text(response)
        if not text:
            raise RuntimeError("OpenAI-compatible provider returned empty content")
        return LLMCallResult(
            text=text,
            provider=self.config.provider,
            model=self.config.model,
            raw=response,
        )


def build_llm_call(config: LLMCallConfig) -> LLMCaller:
    provider = normalize_provider(config.provider)
    if provider in OPENAI_COMPAT_PROVIDER_ALIASES:
        return OpenAICall(config)
    raise RuntimeError(f"Unsupported LM provider={config.provider}.")


def load_api_key(provider: str | None = None) -> str:
    resolved_provider = normalize_provider(provider or infer_provider_from_env())
    if resolved_provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")
        return key
    if resolved_provider == "google":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
        return key
    if resolved_provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        return key
    raise RuntimeError(f"Unsupported LM provider={provider}.")


def parse_json_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                value, end = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise
