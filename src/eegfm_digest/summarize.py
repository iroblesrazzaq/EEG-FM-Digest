from __future__ import annotations

import json
from typing import Any

from .llm_gemini import GeminiClient, parse_json_text
from .triage import validate_json

TAG_TAXONOMY: dict[str, list[str]] = {
    "paper_type": ["eeg-fm", "post-training", "benchmark", "survey"],
    "backbone": ["transformer", "mamba-ssm", "moe", "diffusion"],
    "objective": [
        "masked-reconstruction",
        "autoregressive",
        "contrastive",
        "discrete-code-prediction",
    ],
    "tokenization": ["time-patch", "latent-tokens", "discrete-tokens"],
    "topology": ["fixed-montage", "channel-flexible", "topology-agnostic"],
}


def _summary_triage_payload(triage: dict[str, Any]) -> dict[str, Any]:
    reasons = triage.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "decision": triage.get("decision", "reject"),
        "confidence": float(triage.get("confidence", 0.0)),
        "reasons": reasons,
    }


def _base_payload(paper: dict[str, Any], triage: dict[str, Any]) -> dict[str, Any]:
    return {
        "arxiv_id_base": paper["arxiv_id_base"],
        "title": paper["title"],
        "published_date": paper["published"][:10],
        "categories": paper["categories"],
        "abstract": paper["summary"],
        "triage": _summary_triage_payload(triage),
        "allowed_tags": TAG_TAXONOMY,
    }


def _render_prompt(prompt_template: str, payload: dict[str, Any]) -> str:
    return prompt_template.replace("{{INPUT_JSON}}", json.dumps(payload, ensure_ascii=False))


def _count_tokens_or_none(llm: GeminiClient, prompt: str) -> int | None:
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
    llm: GeminiClient,
    max_input_tokens: int,
) -> tuple[dict[str, Any], str]:
    base = _base_payload(paper, triage)
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


def summarize_paper(
    paper: dict[str, Any],
    triage: dict[str, Any],
    raw_fulltext: str,
    fulltext_slices: dict[str, str],
    used_fulltext: bool,
    notes: str,
    llm: GeminiClient,
    prompt_template: str,
    repair_template: str,
    schema: dict[str, Any],
    max_input_tokens: int,
) -> dict[str, Any]:
    payload, mode_notes = _select_payload(
        paper=paper,
        triage=triage,
        raw_fulltext=raw_fulltext,
        fulltext_slices=fulltext_slices,
        prompt_template=prompt_template,
        llm=llm,
        max_input_tokens=max_input_tokens,
    )
    merged_notes = f"{notes};{mode_notes}" if notes else mode_notes
    prompt = _render_prompt(prompt_template, payload)
    raw = llm.generate(prompt, schema=schema)
    try:
        data = parse_json_text(raw)
        data["used_fulltext"] = used_fulltext
        data["notes"] = merged_notes
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
            data["notes"] = merged_notes
            validate_json(data, schema)
            return data
        except Exception:
            # deterministic fallback with schema-compatible defaults
            return {
                "arxiv_id_base": paper["arxiv_id_base"],
                "title": paper["title"],
                "published_date": paper["published"][:10],
                "categories": paper["categories"],
                "paper_type": "other",
                "one_liner": "Summary unavailable due to JSON validation failure.",
                "detailed_summary": "Unable to produce a reliable multi-sentence summary due to JSON validation failure.",
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
                "tags": {
                    "paper_type": [],
                    "backbone": [],
                    "objective": [],
                    "tokenization": [],
                    "topology": [],
                },
                "limitations": ["unknown", "summary_json_error"],
                "used_fulltext": used_fulltext,
                "notes": f"{merged_notes};summary_json_error",
            }
