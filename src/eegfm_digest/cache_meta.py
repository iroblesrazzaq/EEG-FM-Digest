from __future__ import annotations

import hashlib
import json
from typing import Any

TRIAGE_STAGE_LOGIC_VERSION = "20260325-1"
SUMMARY_STAGE_LOGIC_VERSION = "20260325-1"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_json(payload: Any) -> str:
    return _hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def build_stage_descriptor(
    *,
    stage: str,
    provider: str,
    model: str,
    prompt_template: str,
    repair_template: str,
    schema: dict[str, Any],
    stage_logic_version: str,
) -> dict[str, Any]:
    prompt_hash = _hash_text(prompt_template)
    repair_prompt_hash = _hash_text(repair_template)
    schema_hash = _hash_json(schema)
    cache_version = _hash_json(
        {
            "stage": stage,
            "provider": provider,
            "model": model,
            "prompt_hash": prompt_hash,
            "repair_prompt_hash": repair_prompt_hash,
            "schema_hash": schema_hash,
            "stage_logic_version": stage_logic_version,
        }
    )
    return {
        "stage": stage,
        "provider": provider,
        "model": model,
        "cache_version": cache_version,
        "prompt_hash": prompt_hash,
        "repair_prompt_hash": repair_prompt_hash,
        "schema_hash": schema_hash,
        "stage_logic_version": stage_logic_version,
    }


def build_stage_metadata(
    descriptor: dict[str, Any],
    *,
    repair_used: bool,
    updated_at_source: str | None = None,
) -> dict[str, Any]:
    metadata = dict(descriptor)
    metadata["repair_used"] = bool(repair_used)
    if updated_at_source:
        metadata["updated_at_source"] = updated_at_source
    return metadata


def is_cache_current(metadata: dict[str, Any] | None, cache_version: str) -> bool:
    if not isinstance(metadata, dict):
        return False
    return str(metadata.get("cache_version", "")).strip() == cache_version
