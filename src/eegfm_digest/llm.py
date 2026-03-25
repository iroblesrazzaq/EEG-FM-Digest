from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


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


class OpenAICall:
    """OpenAI-compatible chat-completions wrapper.

    This is used with OpenRouter today by setting `base_url` to the OpenRouter API.
    """

    def __init__(self, config: LLMCallConfig):
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing `openai` package. Install project dependencies first.") from exc

        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
        )

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()

    def count_tokens(self, content: str) -> int:
        # Approximate token count for routing fulltext vs slices.
        return max(1, len(content) // 4)

    def _extract_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
            return "".join(parts).strip()
        return ""

    def call(self, prompt: str, schema: dict[str, Any] | None = None) -> LLMCallResult:
        req: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if schema is not None:
            req["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**req)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            response = getattr(exc, "response", None)
            if status_code is None and response is not None:
                status_code = getattr(response, "status_code", None)
            if status_code in {402, 429}:
                detail = ""
                if response is not None:
                    detail = str(getattr(response, "text", "") or "")[:220]
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
    provider = config.provider.strip().lower()
    if provider in {"openai", "openrouter"}:
        return OpenAICall(config)
    raise RuntimeError(f"Unsupported LM provider={config.provider}.")


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return key


def parse_json_text(text: str) -> dict[str, Any]:
    return json.loads(text)
