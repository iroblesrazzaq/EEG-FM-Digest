from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    temperature: float
    max_output_tokens: int


def _extract_text(resp: Any) -> str:
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates"):
        parts: list[str] = []
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            for p in getattr(content, "parts", []) or []:
                txt = getattr(p, "text", None)
                if txt:
                    parts.append(txt)
        return "\n".join(parts)
    return str(resp)


class GeminiClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        from google import genai

        self._client = genai.Client(api_key=config.api_key)

    def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        cfg: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = schema
        resp = self._client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=cfg,
        )
        return _extract_text(resp).strip()


def load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
    return key


def parse_json_text(text: str) -> dict[str, Any]:
    return json.loads(text)
