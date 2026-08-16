"""PDF download/extract helpers shared by pipeline and batch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .pdf import bounded_text, slice_paper_text
from .row_views import empty_pdf_state


@dataclass(frozen=True)
class PdfTextResult:
    raw_text: str
    pdf_state: dict[str, Any]
    notes: str


@dataclass(frozen=True)
class SummaryTextInputs:
    raw_text: str
    slices: dict[str, str]
    used_fulltext: bool
    notes: str


_EMPTY_SLICES = {
    "abstract": "",
    "introduction": "",
    "methods": "",
    "results": "",
    "conclusion": "",
    "excerpt": "",
}


def summary_inputs_from_pdf_result(
    paper: dict[str, Any],
    pdf_result: PdfTextResult,
    *,
    head_chars: int,
    excerpt_chars: int,
    tail_chars: int,
) -> SummaryTextInputs | None:
    """Build summarizer inputs from extracted PDF text, or abstract-only fallback.

    Returns ``None`` for ``--no-pdf`` so callers skip summarization entirely.
    Missing/failed/empty PDFs still produce inputs with ``used_fulltext=False``.
    """
    if pdf_result.notes == "summary_skipped:no_pdf_mode":
        return None

    raw = (pdf_result.raw_text or "").strip()
    if raw:
        bounded = bounded_text(pdf_result.raw_text, head_chars, tail_chars)
        return SummaryTextInputs(
            raw_text=bounded,
            slices=slice_paper_text(
                bounded,
                excerpt_chars=excerpt_chars,
                tail_chars=min(tail_chars, excerpt_chars),
            ),
            used_fulltext=True,
            notes=pdf_result.notes,
        )

    abstract = str(paper.get("summary") or "").strip()
    slices = dict(_EMPTY_SLICES)
    slices["abstract"] = abstract
    slices["excerpt"] = abstract
    return SummaryTextInputs(
        raw_text=abstract,
        slices=slices,
        used_fulltext=False,
        notes=f"{pdf_result.notes};abstract_only_fallback",
    )


def summary_used_fulltext(summary: dict[str, Any] | None) -> bool:
    """True when a stored summary was produced from extracted PDF text."""
    if not isinstance(summary, dict):
        return False
    return bool(summary.get("used_fulltext"))


def prepare_pdf_and_text(
    paper: dict[str, Any],
    month_out: Path,
    cfg: Config,
    *,
    no_pdf: bool = False,
) -> PdfTextResult:
    arxiv_id_base = paper["arxiv_id_base"]
    pdf_state: dict[str, Any] = empty_pdf_state()
    raw_text = ""
    notes = "summary_not_attempted"

    if no_pdf:
        return PdfTextResult(
            raw_text="",
            pdf_state={
                "downloaded": False,
                "pdf_path": None,
                "text_path": None,
                "extract_meta": {"error": "no_pdf_mode"},
            },
            notes="summary_skipped:no_pdf_mode",
        )

    if not paper.get("links", {}).get("pdf"):
        return PdfTextResult(
            raw_text="",
            pdf_state={
                "downloaded": False,
                "pdf_path": None,
                "text_path": None,
                "extract_meta": {"error": "missing_pdf_link"},
            },
            notes="summary_skipped:missing_pdf_link",
        )

    pdf_path = month_out / "pdfs" / f"{arxiv_id_base}.pdf"
    txt_path = month_out / "text" / f"{arxiv_id_base}.txt"
    try:
        from . import pipeline

        pipeline.download_pdf(paper["links"]["pdf"], pdf_path, cfg.pdf_rate_limit_seconds)
        meta = pipeline.extract_text(pdf_path, txt_path)
        raw_text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        pdf_state = {
            "downloaded": True,
            "pdf_path": str(pdf_path),
            "text_path": str(txt_path),
            "extract_meta": meta,
        }
        notes = json.dumps(meta, sort_keys=True)
    except Exception as exc:
        pdf_state = {
            "downloaded": False,
            "pdf_path": str(pdf_path),
            "text_path": str(txt_path),
            "extract_meta": {"error": f"download_or_extract_failed:{type(exc).__name__}"},
        }
        notes = f"summary_skipped:pdf_failed:{type(exc).__name__}"

    return PdfTextResult(raw_text=raw_text, pdf_state=pdf_state, notes=notes)
