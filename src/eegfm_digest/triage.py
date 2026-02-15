from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from .llm_gemini import GeminiClient, parse_json_text


class SchemaValidationError(RuntimeError):
    pass


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_json(data: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        validate(data, schema)
    except ValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc


def apply_decision_policy(decision: dict[str, Any]) -> dict[str, Any]:
    conf = float(decision.get("confidence", 0.0))
    if decision.get("is_eeg_related") and decision.get("is_foundation_model_related") and conf >= 0.6:
        decision["decision"] = "accept"
        decision["borderline"] = False
    elif conf >= 0.35:
        decision["decision"] = "borderline"
        decision["borderline"] = True
    else:
        decision["decision"] = "reject"
    return decision


def triage_paper(
    paper: dict[str, Any],
    llm: GeminiClient,
    prompt_template: str,
    repair_template: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    input_payload = {
        "arxiv_id_base": paper["arxiv_id_base"],
        "title": paper["title"],
        "authors": paper["authors"],
        "categories": paper["categories"],
        "published_date": paper["published"][:10],
        "abstract": paper["summary"],
        "links": paper["links"],
    }
    prompt = prompt_template.replace("{{INPUT_JSON}}", json.dumps(input_payload, ensure_ascii=False))
    raw = llm.generate(prompt, schema=schema)
    try:
        data = parse_json_text(raw)
        data = apply_decision_policy(data)
        validate_json(data, schema)
        return data
    except Exception:
        repair_prompt = (
            repair_template.replace("{{SCHEMA_JSON}}", json.dumps(schema, ensure_ascii=False))
            .replace("{{BAD_OUTPUT}}", raw)
        )
        try:
            repaired = llm.generate(repair_prompt, schema=schema)
            data = parse_json_text(repaired)
            data = apply_decision_policy(data)
            validate_json(data, schema)
            return data
        except Exception:
            return {
                "arxiv_id_base": paper["arxiv_id_base"],
                "is_eeg_related": False,
                "is_foundation_model_related": False,
                "borderline": False,
                "paper_type": "other",
                "confidence": 0.0,
                "reasons": ["triage_json_error"],
                "suggested_digest_tags": [],
                "decision": "reject",
            }
