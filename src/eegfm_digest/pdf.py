from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx


def download_pdf(pdf_url: str, out_path: Path, rate_limit_seconds: float) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    with httpx.Client(timeout=60) as client:
        resp = client.get(pdf_url)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    time.sleep(rate_limit_seconds)
    return out_path


def extract_text(pdf_path: Path, text_path: Path) -> dict[str, Any]:
    text_path.parent.mkdir(parents=True, exist_ok=True)
    if text_path.exists():
        txt = text_path.read_text(encoding="utf-8")
        return {"tool": "cached", "pages": None, "chars": len(txt), "error": None}

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        chunks = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(chunks)
        text_path.write_text(text, encoding="utf-8")
        return {"tool": "pypdf", "pages": len(reader.pages), "chars": len(text), "error": None}
    except Exception as exc:
        try:
            from pdfminer.high_level import extract_text as pm_extract_text

            text = pm_extract_text(str(pdf_path))
            text_path.write_text(text, encoding="utf-8")
            return {"tool": "pdfminer", "pages": None, "chars": len(text), "error": f"pypdf_failed:{exc}"}
        except Exception as exc2:
            text_path.write_text("", encoding="utf-8")
            return {"tool": "none", "pages": None, "chars": 0, "error": f"extract_failed:{exc2}"}


def bounded_text(text: str, head_chars: int, tail_chars: int) -> str:
    if len(text) <= head_chars + tail_chars:
        return text
    return text[:head_chars] + "\n\n[...TRUNCATED...]\n\n" + text[-tail_chars:]
