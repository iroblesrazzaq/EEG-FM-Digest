from __future__ import annotations

import json
from typing import Any

from .llm_openai_compat import parse_json_text
from .triage import validate_json


def _summary_triage_payload(triage: dict[str, Any]) -> dict[str, Any]:
    reasons = triage.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "decision": triage.get("decision", "reject"),
        "confidence": float(triage.get("confidence", 0.0)),
        "reasons": reasons,
    }


def _base_payload(
    paper: dict[str, Any],
    triage: dict[str, Any],
    allowed_tags: dict[str, list[str]] | None,
) -> dict[str, Any]:
    payload = {
        "arxiv_id_base": paper["arxiv_id_base"],
        "title": paper["title"],
        "published_date": paper["published"][:10],
        "categories": paper["categories"],
        "abstract": paper["summary"],
        "triage": _summary_triage_payload(triage),
    }
    if isinstance(allowed_tags, dict) and allowed_tags:
        payload["allowed_tags"] = allowed_tags
    return payload


def _render_prompt(prompt_template: str, payload: dict[str, Any]) -> str:
    return prompt_template.replace("{{INPUT_JSON}}", json.dumps(payload, ensure_ascii=False))


def _count_tokens_or_none(llm: Any, prompt: str) -> int | None:
    count_fn = getattr(llm, "count_tokens", None)
    if not callable(count_fn):
        return None
    try:
        return int(count_fn(prompt))
    except Exception:
        return None


def _select_payload(
    paper: dict[str, Any],
    triage: dict[str, Any],
    raw_fulltext: str,
    fulltext_slices: dict[str, str],
    prompt_template: str,
    llm: Any,
    max_input_tokens: int,
    allowed_tags: dict[str, list[str]] | None,
) -> tuple[dict[str, Any], str]:
    base = _base_payload(paper, triage, allowed_tags=allowed_tags)
    if not raw_fulltext.strip():
        return {**base, "fulltext_slices": fulltext_slices}, "input_mode=fulltext_slices;reason=missing_fulltext"

    payload_fulltext = {**base, "fulltext": raw_fulltext}
    prompt_fulltext = _render_prompt(prompt_template, payload_fulltext)
    token_count = _count_tokens_or_none(llm, prompt_fulltext)

    if token_count is not None and token_count <= max_input_tokens:
        return payload_fulltext, f"input_mode=fulltext;prompt_tokens={token_count}"
    if token_count is None:
        return {**base, "fulltext_slices": fulltext_slices}, "input_mode=fulltext_slices;reason=count_tokens_failed"
    return {
        **base,
        "fulltext_slices": fulltext_slices,
    }, f"input_mode=fulltext_slices;reason=fulltext_over_limit;prompt_tokens={token_count};max_tokens={max_input_tokens}"


def _schema_has_key(schema: dict[str, Any], key: str) -> bool:
    props = schema.get("properties", {})
    return isinstance(props, dict) and key in props


def _inject_deterministic_fields(
    out: dict[str, Any],
    schema: dict[str, Any],
    paper: dict[str, Any],
    used_fulltext: bool,
    notes: str,
) -> None:
    deterministic = {
        "arxiv_id_base": paper["arxiv_id_base"],
        "title": paper["title"],
        "published_date": paper["published"][:10],
        "categories": paper["categories"],
        "used_fulltext": bool(used_fulltext),
        "notes": str(notes),
    }
    for key, value in deterministic.items():
        if _schema_has_key(schema, key):
            out[key] = value


def _filter_allowed_tags(
    out: dict[str, Any],
    allowed_tags: dict[str, list[str]] | None,
) -> None:
    if not isinstance(allowed_tags, dict) or not allowed_tags:
        return
    tags = out.get("tags")
    if not isinstance(tags, dict):
        return
    filtered: dict[str, list[str]] = {}
    for category, allowed_values in allowed_tags.items():
        allowed_set = {str(item) for item in allowed_values}
        raw_values = tags.get(category, [])
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        if not isinstance(raw_values, list):
            raw_values = []
        keep: list[str] = []
        for value in raw_values:
            value_str = str(value).strip()
            if value_str in allowed_set and value_str not in keep:
                keep.append(value_str)
            if len(keep) >= 2:
                break
        filtered[str(category)] = keep
    out["tags"] = filtered


def _pick_schema_variant(schema: dict[str, Any]) -> dict[str, Any]:
    if isinstance(schema.get("oneOf"), list) and schema["oneOf"]:
        variants = [v for v in schema["oneOf"] if isinstance(v, dict)]
        for variant in variants:
            t = variant.get("type")
            if t != "null":
                return variant
        return variants[0]
    if isinstance(schema.get("anyOf"), list) and schema["anyOf"]:
        variants = [v for v in schema["anyOf"] if isinstance(v, dict)]
        for variant in variants:
            t = variant.get("type")
            if t != "null":
                return variant
        return variants[0]
    return schema


def _string_default(schema: dict[str, Any]) -> str:
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return str(schema["enum"][0])
    if "const" in schema:
        return str(schema["const"])
    min_len = int(schema.get("minLength", 0) or 0)
    max_len = int(schema.get("maxLength", 0) or 0)
    base = "unknown"
    if min_len > 0:
        reps = (min_len + len(base) - 1) // len(base)
        base = (base * reps)[:max(min_len, len(base))]
    if max_len > 0:
        base = base[:max_len]
    return base


def _numeric_default(schema: dict[str, Any], is_int: bool) -> int | float:
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        value = schema["enum"][0]
        return int(value) if is_int else float(value)
    low = schema.get("minimum")
    if low is None:
        low = schema.get("exclusiveMinimum")
        if low is not None:
            low = float(low) + (1 if is_int else 1e-6)
    value = float(low) if low is not None else 0.0
    if is_int:
        value = int(value)
    high = schema.get("maximum")
    if high is not None and value > float(high):
        value = float(high)
    if is_int:
        high_ex = schema.get("exclusiveMaximum")
        if high_ex is not None and value >= int(float(high_ex)):
            value = int(float(high_ex)) - 1
        return int(value)
    high_ex = schema.get("exclusiveMaximum")
    if high_ex is not None and value >= float(high_ex):
        value = float(high_ex) - 1e-6
    return float(value)


def _schema_default(schema: dict[str, Any]) -> Any:
    schema = _pick_schema_variant(schema)
    if "default" in schema:
        return schema["default"]
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return schema["enum"][0]

    type_value = schema.get("type")
    if isinstance(type_value, list):
        non_null = [t for t in type_value if t != "null"]
        if not non_null:
            return None
        type_value = non_null[0]
    if type_value == "null":
        return None
    if type_value == "boolean":
        return False
    if type_value == "integer":
        return _numeric_default(schema, is_int=True)
    if type_value == "number":
        return _numeric_default(schema, is_int=False)
    if type_value == "string":
        return _string_default(schema)
    if type_value == "array":
        min_items = int(schema.get("minItems", 0) or 0)
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        return [_schema_default(item_schema) for _ in range(max(0, min_items))]
    if type_value == "object" or isinstance(schema.get("properties"), dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required", [])
        if not isinstance(required, list):
            required = []
        out: dict[str, Any] = {}
        for key in required:
            if key in props and isinstance(props[key], dict):
                out[str(key)] = _schema_default(props[key])
            else:
                out[str(key)] = "unknown"
        return out
    return None


def _schema_fallback_output(
    schema: dict[str, Any],
    paper: dict[str, Any],
    used_fulltext: bool,
    notes: str,
) -> dict[str, Any]:
    raw = _schema_default(schema)
    out = raw if isinstance(raw, dict) else {}
    _inject_deterministic_fields(out, schema=schema, paper=paper, used_fulltext=used_fulltext, notes=notes)
    return out


def _normalize_summary_output(
    data: dict[str, Any],
    schema: dict[str, Any],
    paper: dict[str, Any],
    used_fulltext: bool,
    notes: str,
    allowed_tags: dict[str, list[str]] | None,
) -> dict[str, Any]:
    out = dict(data)
    _inject_deterministic_fields(out, schema=schema, paper=paper, used_fulltext=used_fulltext, notes=notes)
    _filter_allowed_tags(out, allowed_tags=allowed_tags)
    return out


def summarize_paper(
    paper: dict[str, Any],
    triage: dict[str, Any],
    raw_fulltext: str,
    fulltext_slices: dict[str, str],
    used_fulltext: bool,
    notes: str,
    llm: Any,
    prompt_template: str,
    repair_template: str,
    schema: dict[str, Any],
    max_input_tokens: int,
    allowed_tags: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    payload, mode_notes = _select_payload(
        paper=paper,
        triage=triage,
        raw_fulltext=raw_fulltext,
        fulltext_slices=fulltext_slices,
        prompt_template=prompt_template,
        llm=llm,
        max_input_tokens=max_input_tokens,
        allowed_tags=allowed_tags,
    )
    merged_notes = f"{notes};{mode_notes}" if notes else mode_notes
    prompt = _render_prompt(prompt_template, payload)
    raw = llm.generate(prompt, schema=schema)
    try:
        data = parse_json_text(raw)
        data = _normalize_summary_output(
            data=data,
            schema=schema,
            paper=paper,
            used_fulltext=used_fulltext,
            notes=merged_notes,
            allowed_tags=allowed_tags,
        )
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
            data = _normalize_summary_output(
                data=data,
                schema=schema,
                paper=paper,
                used_fulltext=used_fulltext,
                notes=merged_notes,
                allowed_tags=allowed_tags,
            )
            validate_json(data, schema)
            return data
        except Exception:
            fallback_notes = f"{merged_notes};summary_json_error" if merged_notes else "summary_json_error"
            fallback = _schema_fallback_output(
                schema=schema,
                paper=paper,
                used_fulltext=used_fulltext,
                notes=fallback_notes,
            )
            validate_json(fallback, schema)
            return fallback


def is_placeholder_summary(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    notes = str(summary.get("notes", ""))
    if "summary_json_error" in notes or "summary_retry_failed" in notes:
        return True
    key_points = summary.get("key_points", [])
    if isinstance(key_points, list):
        normalized_points = [str(item).strip().lower() for item in key_points]
        if normalized_points and all(item in {"", "unknown"} for item in normalized_points):
            return True
    unique = str(summary.get("unique_contribution", "")).strip().lower()
    if unique in {"", "unknown"} and "summary_retry_recovered" not in notes:
        return True
    return False


def should_retry_cached_summary(summary: dict[str, Any] | None) -> bool:
    if not is_placeholder_summary(summary):
        return False
    notes = str((summary or {}).get("notes", ""))
    if "summary_retry_failed" in notes:
        return False
    return True


def mark_summary_retry_failed(summary: dict[str, Any], reason: str) -> dict[str, Any]:
    out = dict(summary)
    notes = str(out.get("notes", "")).strip()
    marker = f"summary_retry_failed;summary_retry_failed_reason={reason}"
    out["notes"] = f"{notes};{marker}" if notes else marker
    return out


def mark_summary_retry_recovered(summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(summary)
    notes = str(out.get("notes", "")).strip()
    marker = "summary_retry_recovered"
    out["notes"] = f"{notes};{marker}" if notes else marker
    return out
