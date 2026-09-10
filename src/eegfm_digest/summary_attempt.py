"""Classify and persist per-paper summary attempt outcomes."""

from __future__ import annotations

from typing import Any

from .llm import LLMRateLimitError

JSON_ERROR_MARKER = "summary_json_error"

OK_CATEGORIES = frozenset({"ok_fulltext", "ok_abstract"})
FAILED_CATEGORIES = frozenset(
    {
        "pdf_download",
        "pdf_extract",
        "llm_rate_limit",
        "llm_invalid_json",
        "llm_other",
        "skipped_no_pdf",
    }
)


def summary_is_json_error(summary: dict[str, Any] | None) -> bool:
    """True when a stored summary is the deterministic JSON-repair placeholder."""
    if not isinstance(summary, dict):
        return False
    notes = str(summary.get("notes") or "")
    if JSON_ERROR_MARKER in notes:
        return True
    limitations = summary.get("limitations")
    return isinstance(limitations, list) and JSON_ERROR_MARKER in limitations


def summary_attempt_category(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("category") or "").strip()


def is_failed_attempt_category(category: str) -> bool:
    return category in FAILED_CATEGORIES


def make_summary_attempt(category: str, error: str | None = None) -> dict[str, Any]:
    return {
        "category": category,
        "error": (error or "").strip() or None,
    }


def _extract_meta_error(pdf_state: dict[str, Any] | None) -> str:
    if not isinstance(pdf_state, dict):
        return ""
    meta = pdf_state.get("extract_meta")
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("error") or "").strip()


def _pdf_failure_category(pdf_state: dict[str, Any] | None, notes: str) -> str | None:
    err = _extract_meta_error(pdf_state)
    notes_l = notes or ""
    if err == "no_pdf_mode" or "summary_skipped:no_pdf_mode" in notes_l:
        return "skipped_no_pdf"
    if err == "missing_pdf_link" or "summary_skipped:missing_pdf_link" in notes_l:
        return "pdf_download"
    if isinstance(pdf_state, dict) and pdf_state.get("downloaded") is True and err:
        return "pdf_extract"
    if err.startswith("download_or_extract_failed") or "summary_skipped:pdf_failed:" in notes_l:
        return "pdf_download"
    if err:
        return "pdf_extract"
    return None


def classify_summary_attempt(
    *,
    summary: dict[str, Any] | None,
    pdf_state: dict[str, Any] | None = None,
    failed: bool = False,
    notes: str = "",
    exc: BaseException | None = None,
    no_pdf: bool = False,
    json_error: bool = False,
) -> dict[str, Any]:
    """Return a ``summary_attempt`` dict with ``category`` and optional ``error``."""
    error_name = type(exc).__name__ if exc is not None else None
    combined_notes = notes
    if isinstance(summary, dict) and not combined_notes:
        combined_notes = str(summary.get("notes") or "")

    if no_pdf:
        return make_summary_attempt("skipped_no_pdf", error_name)

    if json_error or summary_is_json_error(summary):
        return make_summary_attempt("llm_invalid_json", error_name)

    if exc is not None:
        if isinstance(exc, LLMRateLimitError):
            return make_summary_attempt("llm_rate_limit", error_name)
        pdf_cat = _pdf_failure_category(pdf_state, combined_notes)
        if pdf_cat and summary is None:
            return make_summary_attempt(pdf_cat, error_name or _extract_meta_error(pdf_state))
        return make_summary_attempt("llm_other", error_name)

    if summary is None or failed:
        pdf_cat = _pdf_failure_category(pdf_state, combined_notes)
        if pdf_cat:
            return make_summary_attempt(pdf_cat, _extract_meta_error(pdf_state) or error_name)
        return make_summary_attempt("llm_other", error_name)

    if summary.get("used_fulltext"):
        return make_summary_attempt("ok_fulltext")
    return make_summary_attempt("ok_abstract")


def attempt_is_hard_failure(attempt: dict[str, Any] | None) -> bool:
    return is_failed_attempt_category(summary_attempt_category(attempt))


def site_summary_failed_reason(
    *,
    summary: Any,
    row: dict[str, Any],
    fallback: str,
) -> str | None:
    """Reason shown on the month card, or None when the summary is displayable."""
    attempt = row.get("summary_attempt") if isinstance(row, dict) else None
    category = summary_attempt_category(attempt)
    if summary_is_json_error(summary) or category == "llm_invalid_json":
        error = ""
        if isinstance(attempt, dict):
            error = str(attempt.get("error") or "").strip()
        return f"{category or 'llm_invalid_json'}:{error}" if error else (category or "llm_invalid_json")
    if isinstance(summary, dict) and category in OK_CATEGORIES:
        return None
    if isinstance(summary, dict) and not category:
        return None
    if category and is_failed_attempt_category(category):
        error = ""
        if isinstance(attempt, dict):
            error = str(attempt.get("error") or "").strip()
        return f"{category}:{error}" if error else category
    if isinstance(summary, dict):
        return None
    return fallback
