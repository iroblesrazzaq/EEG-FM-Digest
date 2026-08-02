"""Site JSON payload shaping for month pages and manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def safe_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_links(value: Any, arxiv_id_base: str) -> dict[str, str]:
    if not isinstance(value, dict):
        value = {}
    abs_url = str(value.get("abs", "")).strip() or f"https://arxiv.org/abs/{arxiv_id_base}"
    pdf_url = str(value.get("pdf", "")).strip()
    links: dict[str, str] = {"abs": abs_url}
    if pdf_url:
        links["pdf"] = pdf_url
    return links


def safe_triage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    reasons = value.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "decision": str(value.get("decision", "reject")),
        "confidence": safe_float(value.get("confidence", 0.0)),
        "reasons": [str(item) for item in reasons],
    }


def summary_failure_reason(row: dict[str, Any]) -> str:
    pdf = row.get("pdf")
    if isinstance(pdf, dict):
        meta = pdf.get("extract_meta")
        if isinstance(meta, dict):
            err = str(meta.get("error", "")).strip()
            if err:
                return err
    return "summary_unavailable"


def paper_rows_from_backend(backend_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(backend_rows, key=lambda x: (str(x.get("published", "")), str(x.get("arxiv_id_base", "")))):
        arxiv_id_base = str(row.get("arxiv_id_base", "")).strip()
        if not arxiv_id_base:
            continue
        triage = safe_triage(row.get("triage"))
        if triage["decision"] != "accept":
            continue
        summary = row.get("paper_summary")
        rows.append(
            {
                "arxiv_id_base": arxiv_id_base,
                "arxiv_id": str(row.get("arxiv_id", "")).strip(),
                "title": str(row.get("title", "")).strip(),
                "published_date": str(row.get("published", "")).strip()[:10],
                "authors": safe_str_list(row.get("authors")),
                "categories": safe_str_list(row.get("categories")),
                "links": safe_links(row.get("links"), arxiv_id_base),
                "triage": triage,
                "summary": summary if isinstance(summary, dict) else None,
                "summary_failed_reason": None if isinstance(summary, dict) else summary_failure_reason(row),
            }
        )
    return rows


def paper_rows_from_summaries(
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(summaries, key=lambda x: (str(x.get("published_date", "")), str(x.get("arxiv_id_base", "")))):
        arxiv_id_base = str(summary.get("arxiv_id_base", "")).strip()
        if not arxiv_id_base:
            continue
        meta = metadata.get(arxiv_id_base, {}) if isinstance(metadata.get(arxiv_id_base, {}), dict) else {}
        rows.append(
            {
                "arxiv_id_base": arxiv_id_base,
                "arxiv_id": str(meta.get("arxiv_id", "")).strip(),
                "title": str(summary.get("title", "")).strip(),
                "published_date": str(summary.get("published_date", "")).strip(),
                "authors": safe_str_list(meta.get("authors")),
                "categories": safe_str_list(summary.get("categories")),
                "links": safe_links(meta.get("links"), arxiv_id_base),
                "triage": {"decision": "accept", "confidence": 0.0, "reasons": []},
                "summary": summary,
                "summary_failed_reason": None,
            }
        )
    return rows


def paper_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("arxiv_id_base", "")).strip(): row
        for row in rows
        if isinstance(row, dict) and str(row.get("arxiv_id_base", "")).strip()
    }


def resolve_featured_paper_id(
    month: str,
    papers: list[dict[str, Any]],
    featured_paper: Any,
) -> str | None:
    if featured_paper is None:
        return None
    featured_paper_id = str(featured_paper).strip()
    if not featured_paper_id:
        return None
    if featured_paper_id not in paper_map(papers):
        raise RuntimeError(
            f"Featured paper for {month} references missing paper id {featured_paper_id}."
        )
    return featured_paper_id


def featured_payload_from_row(featured_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(featured_row, dict):
        return None
    summary = featured_row.get("summary")
    links = featured_row.get("links", {})
    featured_id = str(featured_row.get("arxiv_id_base", "")).strip()
    title = str(featured_row.get("title", "")).strip()
    if isinstance(summary, dict):
        title = str(summary.get("title", title)).strip()
    if not title:
        title = featured_id
    abs_url = ""
    if isinstance(links, dict):
        abs_url = str(links.get("abs", "")).strip()
    if not abs_url and featured_id:
        abs_url = f"https://arxiv.org/abs/{featured_id}"
    one_liner = ""
    if isinstance(summary, dict):
        one_liner = str(summary.get("one_liner", "")).strip()
    return {
        "arxiv_id_base": featured_id,
        "title": title,
        "one_liner": one_liner,
        "abs_url": abs_url,
    }


def month_payload(
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
    backend_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if backend_rows is not None:
        papers = paper_rows_from_backend(backend_rows)
    else:
        papers = paper_rows_from_summaries(summaries, metadata)
    stats = digest.get("stats", {}) if isinstance(digest.get("stats"), dict) else {}
    top_picks = digest.get("top_picks", []) if isinstance(digest.get("top_picks"), list) else []
    featured_paper_id = resolve_featured_paper_id(
        month,
        papers,
        digest.get("featured_paper"),
    )
    return {
        "month": month,
        "stats": {
            "candidates": safe_int(stats.get("candidates", 0), 0),
            "accepted": safe_int(stats.get("accepted", len(papers)), len(papers)),
            "summarized": safe_int(
                stats.get("summarized", len([p for p in papers if p.get("summary")])),
                len([p for p in papers if p.get("summary")]),
            ),
        },
        "featured_paper_id": featured_paper_id,
        "top_picks": [str(item) for item in top_picks],
        "papers": papers,
    }


def month_label(month: str) -> str:
    try:
        dt = datetime.strptime(month, "%Y-%m")
        return dt.strftime("%B %Y")
    except Exception:
        return month


def month_manifest_item(month_dir: Path) -> dict[str, Any]:
    month = month_dir.name
    payload_path = month_dir / "papers.json"
    month_rev = "missing"
    if payload_path.exists():
        try:
            month_rev = hashlib.sha256(payload_path.read_bytes()).hexdigest()[:16]
        except Exception:
            month_rev = "missing"
    payload: Any = {}
    if payload_path.exists():
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    papers: list[dict[str, Any]] = []
    candidates = 0
    accepted = 0
    summarized = 0
    if isinstance(payload, list):
        papers = [row for row in payload if isinstance(row, dict)]
        accepted = len(papers)
        summarized = len(papers)
        candidates = len(papers)
    elif isinstance(payload, dict):
        paper_rows = payload.get("papers", [])
        if isinstance(paper_rows, list):
            papers = [row for row in paper_rows if isinstance(row, dict)]
        stats = payload.get("stats", {})
        if isinstance(stats, dict):
            candidates = safe_int(stats.get("candidates", 0), 0)
            accepted = safe_int(stats.get("accepted", len(papers)), len(papers))
            summarized = safe_int(
                stats.get("summarized", len([p for p in papers if isinstance(p.get("summary"), dict)])),
                len([p for p in papers if isinstance(p.get("summary"), dict)]),
            )
    if candidates == 0:
        empty_state = "no_candidates"
    elif accepted == 0:
        empty_state = "no_accepts"
    elif summarized == 0:
        empty_state = "no_summaries"
    else:
        empty_state = "has_papers"
    featured_value = None
    if isinstance(payload, dict):
        if "featured_paper_id" in payload:
            featured_value = payload.get("featured_paper_id")
        elif "featured_paper" in payload:
            featured_value = payload.get("featured_paper")
    featured_paper_id = resolve_featured_paper_id(month, papers, featured_value)
    featured_row = paper_map(papers).get(featured_paper_id or "")
    featured = featured_payload_from_row(featured_row)
    return {
        "month": month,
        "month_label": month_label(month),
        "href": f"digest/{month}/index.html",
        "json_path": f"digest/{month}/papers.json",
        "month_rev": month_rev,
        "stats": {
            "candidates": candidates,
            "accepted": accepted,
            "summarized": summarized,
        },
        "empty_state": empty_state,
        "featured": featured,
    }
