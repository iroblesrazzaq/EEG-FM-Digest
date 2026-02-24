from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _safe_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_links(value: Any, arxiv_id_base: str) -> dict[str, str]:
    if not isinstance(value, dict):
        value = {}
    abs_url = str(value.get("abs", "")).strip() or f"https://arxiv.org/abs/{arxiv_id_base}"
    pdf_url = str(value.get("pdf", "")).strip()
    links: dict[str, str] = {"abs": abs_url}
    if pdf_url:
        links["pdf"] = pdf_url
    return links


def _safe_triage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    reasons = value.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "decision": str(value.get("decision", "reject")),
        "confidence": _safe_float(value.get("confidence", 0.0)),
        "reasons": [str(item) for item in reasons],
    }


def _summary_failure_reason(row: dict[str, Any]) -> str:
    pdf = row.get("pdf")
    if isinstance(pdf, dict):
        meta = pdf.get("extract_meta")
        if isinstance(meta, dict):
            err = str(meta.get("error", "")).strip()
            if err:
                return err
    return "summary_unavailable"


def _paper_rows_from_backend(backend_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(backend_rows, key=lambda x: (str(x.get("published", "")), str(x.get("arxiv_id_base", "")))):
        arxiv_id_base = str(row.get("arxiv_id_base", "")).strip()
        if not arxiv_id_base:
            continue
        triage = _safe_triage(row.get("triage"))
        if triage["decision"] != "accept":
            continue
        summary = row.get("paper_summary")
        rows.append(
            {
                "arxiv_id_base": arxiv_id_base,
                "arxiv_id": str(row.get("arxiv_id", "")).strip(),
                "title": str(row.get("title", "")).strip(),
                "published_date": str(row.get("published", "")).strip()[:10],
                "authors": _safe_str_list(row.get("authors")),
                "categories": _safe_str_list(row.get("categories")),
                "links": _safe_links(row.get("links"), arxiv_id_base),
                "triage": triage,
                "summary": summary if isinstance(summary, dict) else None,
                "summary_failed_reason": None if isinstance(summary, dict) else _summary_failure_reason(row),
            }
        )
    return rows


def _paper_rows_from_summaries(
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
                "authors": _safe_str_list(meta.get("authors")),
                "categories": _safe_str_list(summary.get("categories")),
                "links": _safe_links(meta.get("links"), arxiv_id_base),
                "triage": {"decision": "accept", "confidence": 0.0, "reasons": []},
                "summary": summary,
                "summary_failed_reason": None,
            }
        )
    return rows


def _month_payload(
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
    backend_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if backend_rows is not None:
        papers = _paper_rows_from_backend(backend_rows)
    else:
        papers = _paper_rows_from_summaries(summaries, metadata)
    stats = digest.get("stats", {}) if isinstance(digest.get("stats"), dict) else {}
    top_picks = digest.get("top_picks", []) if isinstance(digest.get("top_picks"), list) else []
    return {
        "month": month,
        "stats": {
            "candidates": _safe_int(stats.get("candidates", 0), 0),
            "accepted": _safe_int(stats.get("accepted", len(papers)), len(papers)),
            "summarized": _safe_int(
                stats.get("summarized", len([p for p in papers if p.get("summary")])),
                len([p for p in papers if p.get("summary")]),
            ),
        },
        "top_picks": [str(item) for item in top_picks],
        "papers": papers,
    }


def _about_digest_block() -> str:
    return (
        "<section class='digest-about'>"
        "<h2>About This Digest</h2>"
        "<p>[why i made digest, how it works]</p>"
        "</section>"
    )


def render_month_page(
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
) -> str:
    del summaries, metadata, digest  # Render path is JSON-driven; data loads client-side.
    month_attr = html.escape(month)
    month_json = html.escape(f"../../digest/{month}/papers.json")
    manifest_json = html.escape("../../data/months.json")
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>EEG-FM Digest {html.escape(month)}</title>
<link rel='stylesheet' href='../../assets/style.css'></head>
<body>
  <main id='digest-app' class='container' data-view='month' data-month='{month_attr}' data-manifest-json='{manifest_json}' data-month-json='{month_json}'>
    <div class='header'>
      <h1>EEG Foundation Model Digest — {html.escape(month)}</h1>
      <p class='sub'><a href='../../index.html'>All months</a></p>
    </div>
    {_about_digest_block()}
    <section id='controls' class='controls'></section>
    <p id='results-meta' class='small'></p>
    <section id='results'></section>
  </main>
  <script src='../../assets/site.js'></script>
</body></html>
"""


def render_home_page(months: list[str]) -> str:
    latest = months[0] if months else ""
    latest_link = f"digest/{latest}/index.html" if latest else "#"
    fallback_months = html.escape(json.dumps(months, ensure_ascii=False))
    links = "\n".join(f"<li><a href='digest/{m}/index.html'>{m}</a></li>" for m in months)
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>EEG-FM Digest</title>
<link rel='stylesheet' href='assets/style.css'></head><body>
<main id='digest-app' class='container' data-view='all' data-month='' data-manifest-json='data/months.json' data-fallback-months='{fallback_months}'>
<h1>EEG Foundation Model Digest</h1>
{_about_digest_block()}
<p class='sub'>Latest month: <a href='{latest_link}'>{latest}</a></p>
<section id='controls' class='controls'></section>
<p id='results-meta' class='small'></p>
<section id='results'></section>
<details class='archive-fallback'>
<summary>Archive (fallback links)</summary>
<ul>{links}</ul>
</details>
</main>
<script src='assets/site.js'></script>
</body></html>
"""


def write_month_site(
    docs_dir: Path,
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
    backend_rows: list[dict[str, Any]] | None = None,
) -> None:
    month_dir = docs_dir / "digest" / month
    month_dir.mkdir(parents=True, exist_ok=True)
    (month_dir / "index.html").write_text(
        render_month_page(month, summaries, metadata, digest), encoding="utf-8"
    )
    payload = _month_payload(month, summaries, metadata, digest, backend_rows)
    (month_dir / "papers.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (month_dir / "digest.json").write_text(
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _month_manifest_item(month_dir: Path) -> dict[str, Any]:
    month = month_dir.name
    payload_path = month_dir / "papers.json"
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
            candidates = _safe_int(stats.get("candidates", 0), 0)
            accepted = _safe_int(stats.get("accepted", len(papers)), len(papers))
            summarized = _safe_int(
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
    return {
        "month": month,
        "href": f"digest/{month}/index.html",
        "json_path": f"digest/{month}/papers.json",
        "stats": {
            "candidates": candidates,
            "accepted": accepted,
            "summarized": summarized,
        },
        "empty_state": empty_state,
    }


def update_home(docs_dir: Path) -> None:
    month_dirs = sorted(
        [p for p in (docs_dir / "digest").iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    ) if (docs_dir / "digest").exists() else []
    months = [p.name for p in month_dirs]
    (docs_dir / "index.html").write_text(render_home_page(months), encoding="utf-8")
    manifest = {
        "latest": months[0] if months else None,
        "months": [_month_manifest_item(month_dir) for month_dir in month_dirs],
    }
    data_dir = docs_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "months.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (docs_dir / ".nojekyll").write_text("\n", encoding="utf-8")
