from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .summarize import is_placeholder_summary

_SHORT_BLURB = (
    "This digest serves as a monthly update on the current EEG foundation model literature on arXiv. "
    "We filter with arXiv title and abstract keywords, and a triage LLM to decide on papers that qualify. "
    "Then, we generate a summary of the entire paper with an LLM. "
    "I manually choose the featured paper of the month."
)

_PROCESS_DETAILS_INTRO = (
    "This digest serves as a monthly update on the current EEG foundation model literature. "
    "I built it so I can keep up to date with the latest EEG FM papers, mostly using Codex 5.3. "
    "The process is as follows:"
)

_PROCESS_DETAILS_STEPS = [
    "First we call the arXiv API to retrieve papers with EEG-FM-related terms in their title and abstract.",
    (
        "Then, we use an LLM on the title and abstract to triage all papers returned by the arXiv search. "
        "The model returns a decision (accept, reject, borderline), its confidence, and 2-4 reasons "
        "for its decision."
    ),
    (
        "Next, for all models accepted by the triage LLM, we download the pdf, extract text with PyMuPDF, "
        "and run a summary LLM where we extract a summary, bullet points, unique contribution, and tags."
    ),
]

_PROCESS_DETAILS_FOOTER = (
    "All triage and summary LLM calls through February 2026 use arcee-ai/trinity-large-preview:free."
)

_PROCESS_LIMITATIONS = [
    "Only checks paper on arXiv.",
    "arXiv keyword search may miss papers.",
    "Triage LLM could misclassify a paper.",
    (
        "Summary LLM is not an expert on the literature - one consequence is that it lacks "
        "the expertise to judge important and novel contributions, so it must rely on the paper "
        "to accurately self-identify novelty/importance."
    ),
]


_EEG_KEYWORDS = [
    "eeg",
    "electroencephalograph*",
    "brainwave*",
]

_FM_KEYWORDS_SET_A = [
    '"foundation model"',
    "pretrain",
    "pretrained",
    '"self-supervised"',
    '"self supervised"',
]

_FM_KEYWORDS_SET_B = [
    '"representation learning"',
    "masked",
    "transfer",
    "generaliz*",
]


def _site_dict(site_content: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(site_content, dict):
        return {}
    value = site_content.get(key)
    return value if isinstance(value, dict) else {}


def _site_text(site_content: dict[str, Any] | None, key: str, default: str) -> str:
    if isinstance(site_content, dict):
        value = site_content.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _site_list(site_content: dict[str, Any] | None, key: str, default: list[str]) -> list[str]:
    if isinstance(site_content, dict):
        value = site_content.get(key)
        if isinstance(value, list):
            out = [str(item).strip() for item in value if str(item).strip()]
            if out:
                return out
    return default


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


def _placeholder_failure_reason(summary: dict[str, Any], row: dict[str, Any]) -> str:
    notes = str(summary.get("notes", "")).strip()
    for token in notes.split(";"):
        cleaned = token.strip()
        if cleaned.startswith("summary_retry_failed_reason="):
            reason = cleaned.split("=", 1)[1].strip()
            if reason:
                return reason
    if "summary_retry_failed" in notes:
        return "summary_retry_failed"
    if "summary_json_error" in notes:
        return "summary_json_error"
    return _summary_failure_reason(row)


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
        summary_out = summary if isinstance(summary, dict) else None
        summary_failed_reason = None
        if summary_out is None:
            summary_failed_reason = _summary_failure_reason(row)
        elif is_placeholder_summary(summary_out):
            summary_failed_reason = _placeholder_failure_reason(summary_out, row)
            summary_out = None
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
                "summary": summary_out,
                "summary_failed_reason": summary_failed_reason,
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


def _about_digest_block(
    process_href: str,
    include_process_cta: bool = False,
    site_content: dict[str, Any] | None = None,
) -> str:
    home_cfg = _site_dict(site_content, "home")
    about_blurb = _site_text(home_cfg, "about_blurb", _SHORT_BLURB)
    cta = (
        f"<p class='small'><a href='{html.escape(process_href)}'>Read the detailed process and prompt design</a></p>"
        if include_process_cta
        else ""
    )
    return (
        "<section class='digest-about'>"
        "<h2>About This Digest</h2>"
        f"<p>{html.escape(about_blurb)}</p>"
        f"{cta}"
        "</section>"
    )


def _load_prompt_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return "(prompt unavailable)"


def _keyword_list_html(items: list[str]) -> str:
    values = "".join(f"<li><code>{html.escape(item)}</code></li>" for item in items)
    return f"<ul>{values}</ul>"


def _nav_html(
    home_href: str,
    explore_href: str,
    process_href: str,
    active_tab: str,
    site_content: dict[str, Any] | None = None,
) -> str:
    nav_cfg = _site_dict(site_content, "nav")
    site_title = _site_text(site_content, "title", "EEG Foundation Model Digest")
    tabs = [
        ("home", _site_text(nav_cfg, "home_label", "Monthly Digest"), home_href),
        ("explore", _site_text(nav_cfg, "search_label", "Search"), explore_href),
        ("process", _site_text(nav_cfg, "about_label", "About"), process_href),
    ]
    links = "".join(
        (
            f"<a class='site-nav-link{' active' if key == active_tab else ''}' "
            f"href='{html.escape(href)}'>{html.escape(label)}</a>"
        )
        for key, label, href in tabs
    )
    return (
        "<header class='site-shell'>"
        "<div class='site-shell-inner'>"
        "<div class='site-brand'>"
        f"<p class='site-title'><a class='site-title-link' href='{html.escape(home_href)}'>{html.escape(site_title)}</a></p>"
        "</div>"
        f"<nav class='site-nav'>{links}</nav>"
        "</div>"
        "</header>"
    )


def render_process_page(
    site_content: dict[str, Any] | None = None,
    triage_prompt_path: Path = Path("prompts/triage.md"),
    summary_prompt_path: Path = Path("prompts/summarize.md"),
) -> str:
    process_cfg = _site_dict(site_content, "process")
    process_intro = _site_text(process_cfg, "intro", _PROCESS_DETAILS_INTRO)
    process_steps = _site_list(process_cfg, "steps", _PROCESS_DETAILS_STEPS)
    process_footer = _site_text(process_cfg, "footer", _PROCESS_DETAILS_FOOTER)
    process_limitations = _site_list(process_cfg, "limitations", _PROCESS_LIMITATIONS)
    eeg_terms = _site_list(process_cfg, "eeg_terms", _EEG_KEYWORDS)
    fm_terms_a = _site_list(process_cfg, "fm_terms_a", _FM_KEYWORDS_SET_A)
    fm_terms_b = _site_list(process_cfg, "fm_terms_b", _FM_KEYWORDS_SET_B)
    keyword_subtitle = _site_text(
        process_cfg,
        "keyword_subtitle",
        "Matching requires one EEG term plus one FM term set in title/abstract.",
    )

    step_items = "".join(f"<li>{html.escape(text)}</li>" for text in process_steps)
    limitation_items = "".join(f"<li>{html.escape(text)}</li>" for text in process_limitations)
    triage_prompt = html.escape(_load_prompt_text(triage_prompt_path))
    summary_prompt = html.escape(_load_prompt_text(summary_prompt_path))
    nav = _nav_html(
        "../index.html",
        "../explore/index.html",
        "../process/index.html",
        "process",
        site_content=site_content,
    )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>About</title>
<link rel='stylesheet' href='../assets/style.css'></head><body>
{nav}
<main class='container process-page'>
<h1>About This Digest</h1>
<section class='process-content'>
<p>{html.escape(process_intro)}</p>
<ul>{step_items}</ul>
<p>{html.escape(process_footer)}</p>
<h2>Limitations</h2>
<ul>{limitation_items}</ul>
<h2>arXiv Retrieval Keywords</h2>
<section class='prompt-details keyword-details'>
<p class='small'>{html.escape(keyword_subtitle)}</p>
<div class='keyword-grid'>
<section>
<p><strong>EEG term set</strong> (used in both queries)</p>
{_keyword_list_html(eeg_terms)}
</section>
<section>
<p><strong>FM term set A</strong></p>
{_keyword_list_html(fm_terms_a)}
</section>
<section>
<p><strong>FM term set B</strong></p>
{_keyword_list_html(fm_terms_b)}
</section>
</div>
</section>
<h2>LLM Prompts</h2>
<p>These are the full prompts used for each stage.</p>
<section class='prompt-details'>
<p><strong>Triage prompt</strong> (<code>prompts/triage.md</code>)</p>
<pre class='prompt-block'>{triage_prompt}</pre>
</section>
<section class='prompt-details'>
<p><strong>Summary prompt</strong> (<code>prompts/summarize.md</code>)</p>
<pre class='prompt-block'>{summary_prompt}</pre>
</section>
</section>
</main>
</body></html>
"""


def _month_label(month: str) -> str:
    try:
        dt = datetime.strptime(month, "%Y-%m")
        return dt.strftime("%B %Y")
    except Exception:
        return month


def render_month_page(
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
    site_content: dict[str, Any] | None = None,
) -> str:
    del summaries, metadata, digest  # Render path is JSON-driven; data loads client-side.
    month_attr = html.escape(month)
    month_json = html.escape(f"../../digest/{month}/papers.json")
    manifest_json = html.escape("../../data/months.json")
    month_title = html.escape(_month_label(month))
    nav = _nav_html(
        "../../index.html",
        "../../explore/index.html",
        "../../process/index.html",
        "home",
        site_content=site_content,
    )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>EEG-FM Digest {html.escape(month)}</title>
<link rel='stylesheet' href='../../assets/style.css'></head>
<body>
  {nav}
  <main id='digest-app' class='container' data-view='month' data-month='{month_attr}' data-manifest-json='{manifest_json}' data-month-json='{month_json}'>
    <section class='hero-banner month-hero'>
      <p class='hero-kicker'>Monthly Digest</p>
      <h1>{month_title} Digest</h1>
      <p class='sub'>Accepted EEG-FM papers and summaries for this month.</p>
    </section>
    <section id='controls' class='controls'></section>
    <p id='results-meta' class='small'></p>
    <section id='results'></section>
  </main>
  <script src='../../assets/site.js'></script>
</body></html>
"""


def render_home_page(months: list[str], site_content: dict[str, Any] | None = None) -> str:
    fallback_months = html.escape(json.dumps(months, ensure_ascii=False))
    nav = _nav_html(
        "index.html",
        "explore/index.html",
        "process/index.html",
        "home",
        site_content=site_content,
    )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>EEG-FM Digest</title>
<link rel='stylesheet' href='assets/style.css'></head><body>
{nav}
<main id='digest-app' class='container' data-view='home' data-month='' data-manifest-json='data/months.json' data-fallback-months='{fallback_months}'>
{_about_digest_block("process/index.html", include_process_cta=False, site_content=site_content)}
<section id='home-controls' class='controls'></section>
<section id='home-results'></section>
</main>
<script src='assets/site.js'></script>
</body></html>
"""


def render_explore_page(months: list[str], site_content: dict[str, Any] | None = None) -> str:
    fallback_months = html.escape(json.dumps(months, ensure_ascii=False))
    nav = _nav_html(
        "../index.html",
        "../explore/index.html",
        "../process/index.html",
        "explore",
        site_content=site_content,
    )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Search</title>
<link rel='stylesheet' href='../assets/style.css'></head><body>
{nav}
<main id='digest-app' class='container' data-view='explore' data-month='' data-manifest-json='../data/months.json' data-fallback-months='{fallback_months}'>
<h1>Search</h1>
<section id='controls' class='controls'></section>
<p id='results-meta' class='small'></p>
<section id='results'></section>
</main>
<script src='../assets/site.js'></script>
</body></html>
"""


def write_month_site(
    docs_dir: Path,
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
    backend_rows: list[dict[str, Any]] | None = None,
    site_content: dict[str, Any] | None = None,
) -> None:
    month_dir = docs_dir / "digest" / month
    month_dir.mkdir(parents=True, exist_ok=True)
    (month_dir / "index.html").write_text(
        render_month_page(month, summaries, metadata, digest, site_content=site_content), encoding="utf-8"
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
    top_picks: list[str] = []
    if isinstance(payload, dict):
        picks = payload.get("top_picks", [])
        if isinstance(picks, list):
            top_picks = [str(item) for item in picks if str(item).strip()]

    featured_row: dict[str, Any] | None = None
    if top_picks:
        paper_map = {str(row.get("arxiv_id_base", "")): row for row in papers if isinstance(row, dict)}
        for paper_id in top_picks:
            row = paper_map.get(paper_id)
            if isinstance(row, dict):
                featured_row = row
                if isinstance(row.get("summary"), dict):
                    break
    if featured_row is None:
        for row in papers:
            if isinstance(row.get("summary"), dict):
                featured_row = row
                break
    if featured_row is None and papers:
        featured_row = papers[0]

    featured: dict[str, Any] | None = None
    if isinstance(featured_row, dict):
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
        featured = {
            "arxiv_id_base": featured_id,
            "title": title,
            "one_liner": one_liner,
            "abs_url": abs_url,
        }
    return {
        "month": month,
        "month_label": _month_label(month),
        "href": f"digest/{month}/index.html",
        "json_path": f"digest/{month}/papers.json",
        "stats": {
            "candidates": candidates,
            "accepted": accepted,
            "summarized": summarized,
        },
        "empty_state": empty_state,
        "featured": featured,
    }


def update_home(
    docs_dir: Path,
    site_content: dict[str, Any] | None = None,
    triage_prompt_path: Path = Path("prompts/triage.md"),
    summary_prompt_path: Path = Path("prompts/summarize.md"),
) -> None:
    month_dirs = sorted(
        [p for p in (docs_dir / "digest").iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    ) if (docs_dir / "digest").exists() else []
    months = [p.name for p in month_dirs]
    (docs_dir / "index.html").write_text(render_home_page(months, site_content=site_content), encoding="utf-8")
    explore_dir = docs_dir / "explore"
    explore_dir.mkdir(parents=True, exist_ok=True)
    (explore_dir / "index.html").write_text(
        render_explore_page(months, site_content=site_content), encoding="utf-8"
    )
    process_dir = docs_dir / "process"
    process_dir.mkdir(parents=True, exist_ok=True)
    (process_dir / "index.html").write_text(
        render_process_page(
            site_content=site_content,
            triage_prompt_path=triage_prompt_path,
            summary_prompt_path=summary_prompt_path,
        ),
        encoding="utf-8",
    )
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
