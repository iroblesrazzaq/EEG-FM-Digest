from __future__ import annotations

import json
from typing import Any

from .llm_gemini import GeminiClient, parse_json_text
from .triage import validate_json


def summarize_paper(
    paper: dict[str, Any],
    triage: dict[str, Any],
    text_window: str,
    used_fulltext: bool,
    notes: str,
    llm: GeminiClient,
    prompt_template: str,
    repair_template: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "arxiv_id_base": paper["arxiv_id_base"],
        "title": paper["title"],
        "published_date": paper["published"][:10],
        "categories": paper["categories"],
        "abstract": paper["summary"],
        "triage": triage,
        "fulltext": text_window,
    }
    prompt = prompt_template.replace("{{INPUT_JSON}}", json.dumps(payload, ensure_ascii=False))
    raw = llm.generate(prompt, schema=schema)
    try:
        data = parse_json_text(raw)
        data["used_fulltext"] = used_fulltext
        data["notes"] = notes
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
            data["used_fulltext"] = used_fulltext
            data["notes"] = notes
            validate_json(data, schema)
            return data
        except Exception:
            # deterministic fallback with schema-compatible defaults
            return {
                "arxiv_id_base": paper["arxiv_id_base"],
                "title": paper["title"],
                "published_date": paper["published"][:10],
                "categories": paper["categories"],
                "paper_type": triage.get("paper_type", "other"),
                "one_liner": "Summary unavailable due to JSON validation failure.",
                "unique_contribution": "unknown",
                "key_points": ["unknown", "unknown", "unknown"],
                "data_scale": {"datasets": [], "subjects": None, "eeg_hours": None, "channels": None},
                "method": {
                    "architecture": None,
                    "objective": None,
                    "pretraining": None,
                    "finetuning": None,
                },
                "evaluation": {"tasks": [], "benchmarks": [], "headline_results": []},
                "open_source": {"code_url": None, "weights_url": None, "license": None},
                "limitations": ["unknown", "summary_json_error"],
                "digest_tags": triage.get("suggested_digest_tags", [])[:15] or ["unknown", "eeg", "foundation-model"],
                "used_fulltext": used_fulltext,
                "notes": f"{notes}; summary_json_error",
            }
