from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from .llm_openai_compat import parse_json_text


class SchemaValidationError(RuntimeError):
    pass


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_json(data: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        validate(data, schema)
    except ValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc


def _persisted_triage(arxiv_id_base: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "arxiv_id_base": arxiv_id_base,
        "decision": data["decision"],
        "confidence": float(data["confidence"]),
        "reasons": list(data["reasons"]),
    }


def triage_paper(
    paper: dict[str, Any],
    llm: Any,
    prompt_template: str,
    repair_template: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    prompt = prompt_template.replace("{{TITLE}}", paper["title"]).replace(
        "{{ABSTRACT}}", paper["summary"]
    )
    raw = llm.generate(prompt, schema=schema)
    try:
        data = parse_json_text(raw)
        validate_json(data, schema)
        return _persisted_triage(paper["arxiv_id_base"], data)
    except Exception:
        repair_prompt = (
            repair_template.replace("{{SCHEMA_JSON}}", json.dumps(schema, ensure_ascii=False))
            .replace("{{BAD_OUTPUT}}", raw)
        )
        try:
            repaired = llm.generate(repair_prompt, schema=schema)
            data = parse_json_text(repaired)
            validate_json(data, schema)
            return _persisted_triage(paper["arxiv_id_base"], data)
        except Exception:
            return {
                "arxiv_id_base": paper["arxiv_id_base"],
                "decision": "reject",
                "confidence": 0.0,
                "reasons": ["triage_json_error", "insufficient_valid_output"],
            }
