from __future__ import annotations

import re
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

    errors: list[str] = []

    try:
        import pymupdf

        doc = pymupdf.open(str(pdf_path))
        try:
            pages = list(doc)
            chunks = [(getattr(page, "get_text")() if hasattr(page, "get_text") else "") or "" for page in pages]
            text = "\n".join(chunks)
        finally:
            close = getattr(doc, "close", None)
            if callable(close):
                close()
        text_path.write_text(text, encoding="utf-8")
        return {"tool": "pymupdf", "pages": len(pages), "chars": len(text), "error": None}
    except Exception as exc:
        errors.append(f"pymupdf_failed:{exc}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        chunks = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(chunks)
        text_path.write_text(text, encoding="utf-8")
        return {
            "tool": "pypdf",
            "pages": len(reader.pages),
            "chars": len(text),
            "error": "; ".join(errors) if errors else None,
        }
    except Exception as exc:
        errors.append(f"pypdf_failed:{exc}")
        try:
            from pdfminer.high_level import extract_text as pm_extract_text

            text = pm_extract_text(str(pdf_path))
            text_path.write_text(text, encoding="utf-8")
            return {
                "tool": "pdfminer",
                "pages": None,
                "chars": len(text),
                "error": "; ".join(errors) if errors else None,
            }
        except Exception as exc2:
            errors.append(f"pdfminer_failed:{exc2}")
            text_path.write_text("", encoding="utf-8")
            return {"tool": "none", "pages": None, "chars": 0, "error": "; ".join(errors)}


def bounded_text(text: str, head_chars: int, tail_chars: int) -> str:
    if len(text) <= head_chars + tail_chars:
        return text
    return text[:head_chars] + "\n\n[...TRUNCATED...]\n\n" + text[-tail_chars:]


_HEADING_PATTERNS: dict[str, tuple[str, ...]] = {
    "abstract": (r"^\s*abstract\s*$",),
    "introduction": (
        r"^\s*introduction\s*$",
        r"^\s*\d+(\.\d+)*\s+introduction\s*$",
    ),
    "methods": (
        r"^\s*(methods|methodology|approach|materials and methods)\s*$",
        r"^\s*\d+(\.\d+)*\s+(methods|methodology|approach)\s*$",
    ),
    "results": (
        r"^\s*(results|experiments|evaluation)\s*$",
        r"^\s*\d+(\.\d+)*\s+(results|experiments|evaluation)\s*$",
    ),
    "conclusion": (
        r"^\s*(conclusion|conclusions|discussion|concluding remarks)\s*$",
        r"^\s*\d+(\.\d+)*\s+(conclusion|conclusions|discussion)\s*$",
    ),
}


def _normalize_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _matches_heading(section: str, line: str) -> bool:
    normalized = line.strip().lower()
    for pattern in _HEADING_PATTERNS[section]:
        if re.match(pattern, normalized, flags=re.IGNORECASE):
            return True
    return False


def _find_headings(lines: list[str]) -> dict[str, int]:
    headings: dict[str, int] = {}
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line or len(line) > 120:
            continue
        for section in ("abstract", "introduction", "methods", "results", "conclusion"):
            if section not in headings and _matches_heading(section, line):
                headings[section] = idx
                break
    return headings


def _extract_between(
    lines: list[str],
    headings: dict[str, int],
    start_key: str,
    end_keys: tuple[str, ...],
    max_chars: int,
) -> str:
    if start_key not in headings:
        return ""
    start = headings[start_key] + 1
    end_candidates = [headings[k] for k in end_keys if k in headings and headings[k] > headings[start_key]]
    end = min(end_candidates) if end_candidates else len(lines)
    chunk = "\n".join(lines[start:end]).strip()
    if len(chunk) > max_chars:
        chunk = chunk[:max_chars]
    return chunk


def slice_paper_text(text: str, excerpt_chars: int = 18_000, tail_chars: int = 0) -> dict[str, str]:
    normalized = _normalize_extracted_text(text)
    lines = normalized.splitlines()
    headings = _find_headings(lines)
    excerpt = normalized[:excerpt_chars]
    if tail_chars > 0 and len(normalized) > excerpt_chars + tail_chars:
        excerpt = excerpt + "\n\n[...TAIL_EXCERPT...]\n\n" + normalized[-tail_chars:]

    return {
        "abstract": _extract_between(
            lines,
            headings,
            "abstract",
            ("introduction", "methods", "results", "conclusion"),
            max_chars=5_000,
        ),
        "introduction": _extract_between(
            lines,
            headings,
            "introduction",
            ("methods", "results", "conclusion"),
            max_chars=9_000,
        ),
        "methods": _extract_between(
            lines,
            headings,
            "methods",
            ("results", "conclusion"),
            max_chars=12_000,
        ),
        "results": _extract_between(
            lines,
            headings,
            "results",
            ("conclusion",),
            max_chars=12_000,
        ),
        "conclusion": _extract_between(
            lines,
            headings,
            "conclusion",
            (),
            max_chars=7_000,
        ),
        "excerpt": excerpt.strip(),
    }
