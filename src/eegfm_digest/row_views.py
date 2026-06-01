"""Shared row/view helpers for triage and PDF state in pipeline outputs."""

from __future__ import annotations

from typing import Any


def empty_pdf_state() -> dict[str, Any]:
    return {
        "downloaded": False,
        "pdf_path": None,
        "text_path": None,
        "extract_meta": None,
    }


def triage_view(triage: dict[str, Any] | None) -> dict[str, Any]:
    triage = triage or {}
    reasons = triage.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "decision": triage.get("decision", "reject"),
        "confidence": float(triage.get("confidence", 0.0)),
        "reasons": reasons,
    }


def normalize_triage_row(arxiv_id_base: str, result: dict[str, Any]) -> dict[str, Any]:
    reasons = result.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "arxiv_id_base": arxiv_id_base,
        "decision": result.get("decision", "reject"),
        "confidence": float(result.get("confidence", 0.0)),
        "reasons": reasons,
    }
