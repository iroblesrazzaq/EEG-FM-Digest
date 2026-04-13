from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .keywords import load_topic_config

_PROCESS_DETAILS_STEPS = [
    "Query arXiv with a topic-specific recall net built from categories and title/abstract keyword queries.",
    "Run an LLM triage pass on title and abstract only to accept, reject, or mark borderline papers.",
    "Download PDFs only for accepted or borderline papers, extract text, and generate structured summaries.",
    "Render deterministic static HTML pages into docs/ so the archive can be published with GitHub Pages.",
]

_PROCESS_LIMITATIONS = [
    "This pipeline only checks papers available on arXiv.",
    "arXiv keyword search may miss papers.",
    "Triage LLM could misclassify a paper.",
    "Summary quality depends on PDF extraction quality and what the paper states explicitly.",
]

_ASSET_VERSION = "20260225-1"
_FEATURED_PAPERS_PATH = Path("configs/featured_papers.json")
_SITE_CONFIG_PATH = Path("configs/site_config.json")
_DEFAULT_THEME_COLOR = "#1e40af"

_ICON_SVGS = {
    "github": (
        "<svg viewBox='0 0 24 24' aria-hidden='true'>"
        "<path d='M12 2C6.477 2 2 6.489 2 12.018c0 4.424 2.865 8.18 6.839 9.504"
        ".5.092.682-.217.682-.483 0-.237-.009-.866-.014-1.7-2.782.605-3.369-1.344"
        "-3.369-1.344-.454-1.157-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608"
        " 1.003.07 1.531 1.031 1.531 1.031.892 1.53 2.341 1.088 2.91.832.091-.647"
        ".349-1.088.635-1.338-2.221-.252-4.555-1.114-4.555-4.956 0-1.094.39-1.99"
        " 1.029-2.692-.103-.253-.446-1.272.098-2.651 0 0 .84-.269 2.75 1.028A9.564"
        " 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.297 2.748-1.028"
        " 2.748-1.028.546 1.379.203 2.398.1 2.651.64.702 1.027 1.598 1.027 2.692 0"
        " 3.851-2.337 4.701-4.566 4.949.359.309.678.918.678 1.849 0 1.335-.012 2.413"
        "-.012 2.741 0 .269.18.58.688.482A10.022 10.022 0 0 0 22 12.018C22 6.489 17.523"
        " 2 12 2z'/>"
        "</svg>"
    ),
    "website": (
        "<svg viewBox='0 0 24 24' aria-hidden='true'>"
        "<path d='M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm7.93 9h-3.08a15.64 15.64 0 0 0"
        "-1.14-5.01A8.03 8.03 0 0 1 19.93 11zM12 4.06c1.12 1.3 2.05 3.7 2.4 6.94H9.6c.35-3.24"
        " 1.28-5.64 2.4-6.94zM4.07 13h3.08c.1 1.74.5 3.44 1.14 5.01A8.03 8.03 0 0 1 4.07 13zm"
        "3.08-2H4.07a8.03 8.03 0 0 1 4.22-5.01A15.64 15.64 0 0 0 7.15 11zM12 19.94c-1.12-1.3"
        "-2.05-3.7-2.4-6.94h4.8c-.35 3.24-1.28 5.64-2.4 6.94zM15.71 18.01c.64-1.57 1.04-3.27"
        " 1.14-5.01h3.08a8.03 8.03 0 0 1-4.22 5.01z'/>"
        "</svg>"
    ),
    "linkedin": (
        "<svg viewBox='0 0 16 16' aria-hidden='true'>"
        "<path d='M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0"
        " .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943"
        " 12.248V6.169H2.542v7.225h2.401zm-1.2-8.21c.837 0 1.358-.554 1.358-1.248"
        "-.015-.709-.52-1.248-1.342-1.248S2.4 3.227 2.4 3.936c0 .694.521 1.248"
        " 1.327 1.248h.016zm4.908 8.21V9.359c0-.216.016-.432.079-.586.173-.431"
        ".568-.878 1.232-.878.869 0 1.216.663 1.216 1.634v3.865h2.401V9.25c0-2.22"
        "-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54"
        " 5.54 0 0 1 .016-.025V6.169H6.251c.03.678 0 7.225 0 7.225h2.4z'/>"
        "</svg>"
    ),
    "email": (
        "<svg viewBox='0 0 16 16' aria-hidden='true'>"
        "<path d='M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2"
        " 2H2a2 2 0 0 1-2-2V4zm2-.5a.5.5 0 0 0-.5.5v.217l6.5 4.062"
        " 6.5-4.062V4a.5.5 0 0 0-.5-.5H2zm12.5 1.549-4.71 2.944"
        " 4.71 2.97V5.05zM14.247 12l-5.246-3.311-.734.458a.5.5 0 0"
        " 1-.53 0l-.734-.458L1.753 12h12.494zM1.5 10.964l4.71-2.97"
        "-4.71-2.944v5.914z'/>"
        "</svg>"
    ),
}


@dataclass(frozen=True)
class SiteBranding:
    title: str
    author: str
    description: str
    theme_color: str
    links: dict[str, str]


def load_site_config(path: Path = _SITE_CONFIG_PATH) -> SiteBranding:
    payload: dict[str, Any] = {
        "title": "EEG Foundation Model Digest",
        "author": "Ismael Robles-Razzaq",
        "description": "Monthly arXiv digest for EEG-FM papers",
        "theme_color": _DEFAULT_THEME_COLOR,
        "links": {
            "github": "https://github.com/iroblesrazzaq",
            "linkedin": "https://www.linkedin.com/in/ismaelroblesrazzaq",
            "website": "https://iroblesrazzaq.github.io/",
            "email": "ismaelroblesrazzaq@gmail.com",
        },
    }
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid site config at {path}: expected a JSON object.")
        payload.update({key: value for key, value in raw.items() if key != "links"})
        if isinstance(raw.get("links"), dict):
            merged_links = dict(payload["links"])
            merged_links.update(
                {
                    str(key).strip(): str(value).strip()
                    for key, value in raw["links"].items()
                    if str(key).strip() and str(value).strip()
                }
            )
            payload["links"] = merged_links
    return SiteBranding(
        title=str(payload.get("title", "Research Digest")).strip() or "Research Digest",
        author=str(payload.get("author", "")).strip(),
        description=str(payload.get("description", "")).strip(),
        theme_color=_normalize_hex_color(str(payload.get("theme_color", _DEFAULT_THEME_COLOR)).strip()),
        links={
            str(key).strip(): str(value).strip()
            for key, value in dict(payload.get("links", {})).items()
            if str(key).strip() and str(value).strip()
        },
    )


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


def _normalize_hex_color(value: str) -> str:
    color = value.strip().lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in color):
        return _DEFAULT_THEME_COLOR
    return f"#{color.lower()}"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    color = _normalize_hex_color(value).lstrip("#")
    return tuple(int(color[idx:idx + 2], 16) for idx in (0, 2, 4))


def _mix_hex(value: str, other: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    rgb_a = _hex_to_rgb(value)
    rgb_b = _hex_to_rgb(other)
    mixed = tuple(round((1.0 - ratio) * a + ratio * b) for a, b in zip(rgb_a, rgb_b))
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def _theme_style_block(branding: SiteBranding) -> str:
    accent = branding.theme_color
    accent_deep = _mix_hex(accent, "#000000", 0.22)
    accent_soft = _mix_hex(accent, "#ffffff", 0.88)
    return (
        "<style>:root{"
        f"--accent:{html.escape(accent)};"
        f"--accent-deep:{html.escape(accent_deep)};"
        f"--accent-soft:{html.escape(accent_soft)};"
        "}</style>"
    )


def _head_html(page_title: str, stylesheet_href: str, branding: SiteBranding, description: str | None = None) -> str:
    meta_description = html.escape(description or branding.description)
    return (
        "<head>"
        "<meta charset='utf-8'>"
        f"<meta name='description' content='{meta_description}'>"
        f"<meta name='theme-color' content='{html.escape(branding.theme_color)}'>"
        f"<title>{html.escape(page_title)}</title>"
        f"<link rel='stylesheet' href='{html.escape(stylesheet_href)}?v={_ASSET_VERSION}'>"
        f"{_theme_style_block(branding)}"
        "</head>"
    )


def load_featured_paper_overrides(path: Path = _FEATURED_PAPERS_PATH) -> dict[str, str | None]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid featured papers config at {path}: expected object")
    overrides: dict[str, str | None] = {}
    for month, value in raw.items():
        month_key = str(month).strip()
        if not month_key:
            continue
        if value is None:
            overrides[month_key] = None
            continue
        paper_id = str(value).strip()
        if not paper_id:
            raise RuntimeError(f"Invalid featured paper override for {month_key}: expected non-empty arXiv id or null")
        overrides[month_key] = paper_id
    return overrides


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


def _paper_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("arxiv_id_base", "")).strip(): row
        for row in rows
        if isinstance(row, dict) and str(row.get("arxiv_id_base", "")).strip()
    }


def _auto_featured_paper_id(
    papers: list[dict[str, Any]],
    top_picks: list[str],
) -> str | None:
    paper_map = _paper_map(papers)
    for paper_id in top_picks:
        row = paper_map.get(paper_id)
        if isinstance(row, dict):
            return paper_id
    for row in papers:
        if isinstance(row.get("summary"), dict):
            return str(row.get("arxiv_id_base", "")).strip() or None
    if papers:
        return str(papers[0].get("arxiv_id_base", "")).strip() or None
    return None


def _resolve_featured_paper_id(
    month: str,
    papers: list[dict[str, Any]],
    top_picks: list[str],
    featured_overrides: dict[str, str | None],
) -> str | None:
    if month in featured_overrides:
        override = featured_overrides[month]
        if override is None:
            return None
        if override not in _paper_map(papers):
            raise RuntimeError(
                f"Featured paper override for {month} references missing paper id {override}."
            )
        return override
    return _auto_featured_paper_id(papers, top_picks)


def _featured_payload_from_row(featured_row: dict[str, Any] | None) -> dict[str, Any] | None:
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


def _month_payload(
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
    backend_rows: list[dict[str, Any]] | None,
    featured_overrides: dict[str, str | None],
) -> dict[str, Any]:
    if backend_rows is not None:
        papers = _paper_rows_from_backend(backend_rows)
    else:
        papers = _paper_rows_from_summaries(summaries, metadata)
    stats = digest.get("stats", {}) if isinstance(digest.get("stats"), dict) else {}
    top_picks = digest.get("top_picks", []) if isinstance(digest.get("top_picks"), list) else []
    featured_paper_id = _resolve_featured_paper_id(
        month,
        papers,
        [str(item) for item in top_picks],
        featured_overrides,
    )
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
        "featured_paper_id": featured_paper_id,
        "top_picks": [str(item) for item in top_picks],
        "papers": papers,
    }


def _about_digest_block(process_href: str, include_process_cta: bool = False) -> str:
    branding = load_site_config()
    cta = (
        f"<p class='small'><a href='{html.escape(process_href)}'>Read the detailed process and prompt design</a></p>"
        if include_process_cta
        else ""
    )
    description = branding.description or "A monthly research digest built from arXiv retrieval and structured LLM summaries."
    return (
        "<section class='digest-about'>"
        "<h2>About This Digest</h2>"
        f"<p>{html.escape(description)}. The pipeline retrieves candidate papers from arXiv, triages them with an LLM, "
        "summarizes accepted papers from PDFs, and publishes a static archive.</p>"
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


def _header_contact_links_html(branding: SiteBranding) -> str:
    items = [
        ("github", "GitHub", branding.links.get("github", "")),
        ("website", "Website", branding.links.get("website", "")),
        ("linkedin", "LinkedIn", branding.links.get("linkedin", "")),
        ("email", "Email", branding.links.get("email", "")),
    ]
    links: list[str] = []
    for icon_name, aria_label, href in items:
        href = href.strip()
        if not href:
            continue
        if icon_name == "email" and not href.startswith("mailto:"):
            href = f"mailto:{href}"
        icon = _ICON_SVGS.get(icon_name, "")
        if not icon:
            continue
        rel = " rel='noopener noreferrer'" if href.startswith("http") else ""
        links.append(
            f"<a class='contact-link' href='{html.escape(href)}' aria-label='{html.escape(aria_label)}'"
            f" title='{html.escape(aria_label)}'{rel}>{icon}<span class='sr-only'>{html.escape(aria_label)}</span></a>"
        )
    return "".join(links)


def _nav_html(
    home_href: str,
    explore_href: str,
    process_href: str,
    active_tab: str,
    branding: SiteBranding,
) -> str:
    tabs = [
        ("home", "Monthly Digest", home_href),
        ("explore", "Search", explore_href),
        ("process", "About", process_href),
    ]
    links = "".join(
        (
            f"<a class='site-nav-link{' active' if key == active_tab else ''}' "
            f"href='{html.escape(href)}'>{html.escape(label)}</a>"
        )
        for key, label, href in tabs
    )
    github_href = branding.links.get("github", "").strip()
    github_link = ""
    if github_href:
        github_link = (
            f"<a class='site-nav-link site-nav-link-repo' href='{html.escape(github_href)}' "
            "rel='noopener noreferrer' target='_blank'>GitHub</a>"
        )
    contacts = _header_contact_links_html(branding)
    return (
        "<header class='site-shell'>"
        "<div class='site-shell-inner'>"
        "<div class='site-shell-top'>"
        "<div class='site-brand'>"
        f"<p class='site-title'><a class='site-title-link' href='{html.escape(home_href)}'>{html.escape(branding.title)}</a></p>"
        "</div>"
        f"<nav class='site-nav'>{links}{github_link}</nav>"
        "</div>"
        "<div class='site-shell-meta'>"
        f"<p class='site-byline'>by <strong>{html.escape(branding.author)}</strong></p>"
        f"<div class='site-contact-links'>{contacts}</div>"
        "</div>"
        "</div>"
        "</header>"
    )


def render_process_page() -> str:
    branding = load_site_config()
    topic = load_topic_config()
    step_items = "".join(f"<li>{html.escape(text)}</li>" for text in _PROCESS_DETAILS_STEPS)
    limitation_items = "".join(f"<li>{html.escape(text)}</li>" for text in _PROCESS_LIMITATIONS)
    triage_prompt = html.escape(_load_prompt_text(Path("prompts/triage.md")))
    summary_prompt = html.escape(_load_prompt_text(Path("prompts/summarize.md")))
    nav = _nav_html("../index.html", "../explore/index.html", "../process/index.html", "process", branding)
    category_items = _keyword_list_html(list(topic.categories))
    query_items = _keyword_list_html(list(topic.queries))
    return f"""<!doctype html>
<html>{_head_html(_tab_title(branding, "About"), "../assets/style.css", branding)}<body>
{nav}
<main class='container process-page'>
<h1>About This Digest</h1>
<section class='process-content'>
<p>{html.escape(branding.description)}. The current topic definition is loaded from <code>configs/topics/</code>.</p>
<ul>{step_items}</ul>
<p>The default model choice is hardcoded in the repo, while the API key is loaded from environment variables. This keeps scheduled runs simple for forks.</p>
<h2>Limitations</h2>
<ul>{limitation_items}</ul>
<h2>Topic Config</h2>
<section class='prompt-details keyword-details'>
<p class='small'>The active topic config drives retrieval categories and query strings.</p>
<div class='keyword-grid'>
<section>
<p><strong>arXiv categories</strong></p>
{category_items}
</section>
<section>
<p><strong>Query strings</strong></p>
{query_items}
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


def _month_tab_label(month: str) -> str:
    try:
        dt = datetime.strptime(month, "%Y-%m")
        return dt.strftime("%b %Y")
    except Exception:
        return month


def _tab_title(branding: SiteBranding, suffix: str | None = None) -> str:
    base = branding.title
    if not suffix:
        return base
    return f"{base} | {suffix}"


def render_month_page(
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
) -> str:
    del summaries, metadata, digest  # Render path is JSON-driven; data loads client-side.
    branding = load_site_config()
    month_attr = html.escape(month)
    month_json = html.escape(f"../../digest/{month}/papers.json")
    manifest_json = html.escape("../../data/months.json")
    month_title = html.escape(_month_label(month))
    month_tab_title = _tab_title(branding, _month_tab_label(month))
    nav = _nav_html("../../index.html", "../../explore/index.html", "../../process/index.html", "home", branding)
    return f"""<!doctype html>
<html>{_head_html(month_tab_title, "../../assets/style.css", branding, f"{branding.description} for {_month_label(month)}.")}
<body>
  {nav}
  <main id='digest-app' class='container' data-view='month' data-month='{month_attr}' data-manifest-json='{manifest_json}' data-month-json='{month_json}'>
    <section class='hero-banner month-hero'>
      <p class='hero-kicker'>Monthly Digest</p>
      <h1>{month_title} Digest</h1>
      <p class='sub'>Accepted papers and summaries for this month.</p>
    </section>
    <section id='controls' class='controls'></section>
    <p id='results-meta' class='small'></p>
    <section id='results'></section>
  </main>
  <script src='../../assets/site.js?v={_ASSET_VERSION}'></script>
</body></html>
"""


def render_home_page(months: list[str]) -> str:
    branding = load_site_config()
    fallback_months = html.escape(json.dumps(months, ensure_ascii=False))
    nav = _nav_html("index.html", "explore/index.html", "process/index.html", "home", branding)
    return f"""<!doctype html>
<html>{_head_html(_tab_title(branding), "assets/style.css", branding)}<body>
{nav}
<main id='digest-app' class='container' data-view='home' data-month='' data-manifest-json='data/months.json' data-fallback-months='{fallback_months}'>
{_about_digest_block("process/index.html", include_process_cta=False)}
<section id='home-controls' class='controls'></section>
<section id='home-results'></section>
</main>
<script src='assets/site.js?v={_ASSET_VERSION}'></script>
</body></html>
"""


def render_explore_page(months: list[str]) -> str:
    branding = load_site_config()
    fallback_months = html.escape(json.dumps(months, ensure_ascii=False))
    nav = _nav_html("../index.html", "../explore/index.html", "../process/index.html", "explore", branding)
    return f"""<!doctype html>
<html>{_head_html(_tab_title(branding, "Search"), "../assets/style.css", branding)}<body>
{nav}
<main id='digest-app' class='container' data-view='explore' data-month='' data-manifest-json='../data/months.json' data-fallback-months='{fallback_months}'>
<h1>Search</h1>
<section id='controls' class='controls'></section>
<p id='results-meta' class='small'></p>
<section id='results'></section>
</main>
<script src='../assets/site.js?v={_ASSET_VERSION}'></script>
</body></html>
"""


def write_month_site(
    docs_dir: Path,
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
    backend_rows: list[dict[str, Any]] | None = None,
    featured_overrides: dict[str, str | None] | None = None,
) -> None:
    month_dir = docs_dir / "digest" / month
    month_dir.mkdir(parents=True, exist_ok=True)
    (month_dir / "index.html").write_text(
        render_month_page(month, summaries, metadata, digest), encoding="utf-8"
    )
    overrides = load_featured_paper_overrides() if featured_overrides is None else featured_overrides
    payload = _month_payload(month, summaries, metadata, digest, backend_rows, overrides)
    (month_dir / "papers.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (month_dir / "digest.json").write_text(
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _month_manifest_item(
    month_dir: Path,
    featured_overrides: dict[str, str | None],
) -> dict[str, Any]:
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

    featured_paper_id = _resolve_featured_paper_id(month, papers, top_picks, featured_overrides)
    featured_row = _paper_map(papers).get(featured_paper_id or "")
    featured = _featured_payload_from_row(featured_row)
    return {
        "month": month,
        "month_label": _month_label(month),
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


def update_home(
    docs_dir: Path,
    featured_overrides: dict[str, str | None] | None = None,
) -> None:
    month_dirs = sorted(
        [p for p in (docs_dir / "digest").iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    ) if (docs_dir / "digest").exists() else []
    overrides = load_featured_paper_overrides() if featured_overrides is None else featured_overrides
    months = [p.name for p in month_dirs]
    (docs_dir / "index.html").write_text(render_home_page(months), encoding="utf-8")
    explore_dir = docs_dir / "explore"
    explore_dir.mkdir(parents=True, exist_ok=True)
    (explore_dir / "index.html").write_text(render_explore_page(months), encoding="utf-8")
    process_dir = docs_dir / "process"
    process_dir.mkdir(parents=True, exist_ok=True)
    (process_dir / "index.html").write_text(render_process_page(), encoding="utf-8")
    manifest = {
        "latest": months[0] if months else None,
        "months": [_month_manifest_item(month_dir, overrides) for month_dir in month_dirs],
    }
    data_dir = docs_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "months.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (docs_dir / ".nojekyll").write_text("\n", encoding="utf-8")
