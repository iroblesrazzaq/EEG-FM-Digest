from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class OpenAICompatConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_output_tokens: int
    request_timeout_seconds: float = 180.0
    token_chars_per_token: int = 4


class RateLimitError(RuntimeError):
    pass


class OpenAICompatClient:
    def __init__(self, config: OpenAICompatConfig):
        self.config = config
        timeout = httpx.Timeout(connect=30.0, read=config.request_timeout_seconds, write=30.0, pool=30.0)
        self._client = httpx.Client(timeout=timeout)
        self._endpoint = config.base_url.rstrip("/") + "/chat/completions"

    def close(self) -> None:
        self._client.close()

    def count_tokens(self, content: str) -> int:
        chars_per_token = max(1, int(self.config.token_chars_per_token))
        return max(1, int(math.ceil(len(content) / chars_per_token)))

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict) and item.get("type") in {"text", None}:
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts).strip()
        return ""

    def _request(self, body: dict[str, Any]) -> httpx.Response:
        response = self._client.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if response.status_code == 429:
            raise RateLimitError(f"LLM rate limit status=429 body={response.text[:300]}")
        if response.status_code >= 500:
            raise RuntimeError(f"LLM server error status={response.status_code} body={response.text[:300]}")
        return response

    def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        base_body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }

        attempted_schema = False
        if schema is not None:
            attempted_schema = True
            body_with_schema = dict(base_body)
            body_with_schema["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
            response = self._request(body_with_schema)
            if response.status_code in {400, 422}:
                attempted_schema = False
            else:
                response.raise_for_status()
                text = self._extract_text(response.json())
                if text:
                    return text
                raise RuntimeError("LLM response missing text content")

        response = self._request(base_body)
        if response.status_code in {400, 422} and attempted_schema:
            # Should not occur because schema failures are handled above, but keep deterministic behavior.
            response = self._request(base_body)
        response.raise_for_status()
        text = self._extract_text(response.json())
        if not text:
            raise RuntimeError("LLM response missing text content")
        return text


def load_api_key(env_var: str) -> str:
    value = str(env_var).strip()
    if not value:
        raise RuntimeError("Missing API key environment variable name.")
    key = str(os.environ.get(value, "")).strip()
    if not key:
        raise RuntimeError(f"Missing API key in environment variable: {value}")
    return key


def parse_json_text(text: str) -> dict[str, Any]:
    return json.loads(text)
