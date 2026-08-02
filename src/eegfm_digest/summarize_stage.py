"""PDF download/extract helpers shared by pipeline and batch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .row_views import empty_pdf_state


@dataclass(frozen=True)
class PdfTextResult:
    raw_text: str
    pdf_state: dict[str, Any]
    notes: str


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
