"""Assemble per-candidate backend row records for JSONL output."""

from __future__ import annotations

from typing import Any

from .row_views import empty_pdf_state, triage_view


def build_backend_rows(
    candidates: list[dict[str, Any]],
    triage_map: dict[str, dict[str, Any]],
    summary_map: dict[str, dict[str, Any] | None],
    pdf_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper in sorted(candidates, key=lambda x: (x["published"], x["arxiv_id_base"])):
        arxiv_id_base = paper["arxiv_id_base"]
        rows.append(
            {
                "arxiv_id": paper["arxiv_id"],
                "arxiv_id_base": arxiv_id_base,
                "version": paper["version"],
                "title": paper["title"],
                "summary": paper["summary"],
                "authors": paper["authors"],
                "categories": paper["categories"],
                "published": paper["published"],
                "updated": paper["updated"],
                "links": paper["links"],
                "triage": triage_view(triage_map.get(arxiv_id_base)),
                "paper_summary": summary_map.get(arxiv_id_base),
                "pdf": pdf_map.get(arxiv_id_base, empty_pdf_state()),
            }
        )
    return rows
