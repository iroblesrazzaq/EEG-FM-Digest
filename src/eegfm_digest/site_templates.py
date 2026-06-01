"""Static HTML templates and copy for the published site."""

from __future__ import annotations

import html
import json
from typing import Any

from .keywords import EEG_KEYWORDS, FM_KEYWORDS_SET_A, FM_KEYWORDS_SET_B
from .resources import prompt_path

SHORT_BLURB = (
    "This digest serves as a monthly update on the current EEG foundation model literature on arXiv. "
    "We filter with arXiv title and abstract keywords, and a triage LLM to decide on papers that qualify. "
    "Then, we generate a summary of the entire paper with an LLM. "
    "I manually choose the featured paper of the month."
)

PROCESS_DETAILS_INTRO = (
    "This digest serves as a monthly update on the current EEG foundation model literature. "
    "I built it so I can keep up to date with the latest EEG FM papers, mostly using Codex 5.3. "
    "The process is as follows:"
)

PROCESS_DETAILS_STEPS = [
    (
        "First we call the arXiv API to retrieve papers with EEG-FM-related terms in their title and abstract. "
        "This yielded me 492 candidate papers."
    ),
    (
        "Then, I use an LLM on the title and abstract to triage all papers returned by the arXiv search. "
        "The model returns a decision (accept, reject, borderline), its confidence, and 2-4 reasons "
        "for its decision. 94 papers passed this step."
    ),
    (
        "Finally, for all models accepted by the triage LLM, we download the pdf, extract text with PyMuPDF, "
        "and run a summary LLM where we extract a summary, bullet points, unique contribution, and tags."
    ),
]

PROCESS_DETAILS_FOOTER = (
    "Current triage and summary LLM calls use gemma-4-31b-it via Google AI Studio. "
    "For all previous papers (2021 - Jan 2026), running this whole process cost ~3 million tokens, so each accepted paper costs "
    "~30,000 tokens (including averaged triage costs for papers that don't pass). Crucially, this digest "
    "excludes models pretrained on data from one specific task and fine-tuned specifically for that same task "
    "- we define an EEG FM as a large model pretrained on EEG data, built with the potential and intention "
    "for broad transfer. I update the digest at least once a month, hopefully every week if I'm diligent."
)

PROCESS_LIMITATIONS = [
    "Only checks paper on arXiv.",
    "arXiv keyword search may miss papers.",
    "Triage LLM could misclassify a paper.",
    (
        "Summary LLM is not an expert on the literature - one consequence is that it lacks "
        "the expertise to judge important and novel contributions, so it must rely on the paper "
        "to accurately self-identify novelty/importance."
    ),
]

AUTHOR_NAME = "Ismael Robles-Razzaq"
GITHUB_PROFILE_URL = "https://github.com/iroblesrazzaq"
PROJECT_REPO_URL = "https://github.com/iroblesrazzaq/EEG-FM-Digest"
PERSONAL_WEBSITE_URL = "https://iroblesrazzaq.github.io/"
LINKEDIN_URL = "https://www.linkedin.com/in/ismaelroblesrazzaq"
EMAIL_ADDRESS = "ismaelroblesrazzaq@gmail.com"
ASSET_VERSION = "20260225-1"
SITE_TAB_TITLE_BASE = "EEG-FM Digest"

ICON_SVGS = {
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


def about_digest_block(process_href: str, include_process_cta: bool = False) -> str:
    cta = (
        f"<p class='small'><a href='{html.escape(process_href)}'>Read the detailed process and prompt design</a></p>"
        if include_process_cta
        else ""
    )
    return (
        "<section class='digest-about'>"
        "<h2>About This Digest</h2>"
        f"<p>{html.escape(SHORT_BLURB)}</p>"
        f"{cta}"
        "</section>"
    )


def load_prompt_text(path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return "(prompt unavailable)"


def keyword_list_html(items: list[str]) -> str:
    values = "".join(f"<li><code>{html.escape(item)}</code></li>" for item in items)
    return f"<ul>{values}</ul>"


def header_contact_links_html() -> str:
    items = [
        ("github", "GitHub profile", GITHUB_PROFILE_URL),
        ("website", "Personal website", PERSONAL_WEBSITE_URL),
        ("linkedin", "LinkedIn profile", LINKEDIN_URL),
        ("email", f"Email {EMAIL_ADDRESS}", f"mailto:{EMAIL_ADDRESS}"),
    ]
    links: list[str] = []
    for icon_name, aria_label, href in items:
        icon = ICON_SVGS.get(icon_name, "")
        if not icon:
            continue
        rel = " rel='noopener noreferrer'" if href.startswith("http") else ""
        links.append(
            f"<a class='contact-link' href='{html.escape(href)}' aria-label='{html.escape(aria_label)}'"
            f" title='{html.escape(aria_label)}'{rel}>{icon}<span class='sr-only'>{html.escape(aria_label)}</span></a>"
        )
    return "".join(links)


def nav_html(
    home_href: str,
    explore_href: str,
    process_href: str,
    active_tab: str,
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
    repo_link = (
        f"<a class='site-nav-link site-nav-link-repo' href='{html.escape(PROJECT_REPO_URL)}' "
        "rel='noopener noreferrer' target='_blank'>GitHub Repo</a>"
    )
    contacts = header_contact_links_html()
    return (
        "<header class='site-shell'>"
        "<div class='site-shell-inner'>"
        "<div class='site-shell-top'>"
        "<div class='site-brand'>"
        f"<p class='site-title'><a class='site-title-link' href='{html.escape(home_href)}'>EEG Foundation Model Digest</a></p>"
        "</div>"
        f"<nav class='site-nav'>{links}{repo_link}</nav>"
        "</div>"
        "<div class='site-shell-meta'>"
        f"<p class='site-byline'>by <strong>{html.escape(AUTHOR_NAME)}</strong></p>"
        f"<div class='site-contact-links'>{contacts}</div>"
        "</div>"
        "</div>"
        "</header>"
    )


def tab_title(suffix: str | None = None) -> str:
    if not suffix:
        return SITE_TAB_TITLE_BASE
    return f"{SITE_TAB_TITLE_BASE} | {suffix}"


def month_tab_label(month: str) -> str:
    from datetime import datetime

    try:
        dt = datetime.strptime(month, "%Y-%m")
        return dt.strftime("%b %Y")
    except Exception:
        return month


def render_process_page() -> str:
    step_items = "".join(f"<li>{html.escape(text)}</li>" for text in PROCESS_DETAILS_STEPS)
    limitation_items = "".join(f"<li>{html.escape(text)}</li>" for text in PROCESS_LIMITATIONS)
    triage_prompt = html.escape(load_prompt_text(prompt_path("triage.md")))
    summary_prompt = html.escape(load_prompt_text(prompt_path("summarize.md")))
    nav = nav_html("../index.html", "../explore/index.html", "../process/index.html", "process")
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(tab_title("About"))}</title>
<link rel='stylesheet' href='../assets/style.css?v={ASSET_VERSION}'></head><body>
{nav}
<main class='container process-page'>
<h1>About This Digest</h1>
<section class='process-content'>
<p>{html.escape(PROCESS_DETAILS_INTRO)}</p>
<ul>{step_items}</ul>
<p>{html.escape(PROCESS_DETAILS_FOOTER)}</p>
<h2>Limitations</h2>
<ul>{limitation_items}</ul>
<h2>arXiv Retrieval Keywords</h2>
<section class='prompt-details keyword-details'>
<p class='small'>Matching requires one EEG term plus one FM term set in title/abstract.</p>
<div class='keyword-grid'>
<section>
<p><strong>EEG term set</strong> (used in both queries)</p>
{keyword_list_html(EEG_KEYWORDS)}
</section>
<section>
<p><strong>FM term set A</strong></p>
{keyword_list_html(FM_KEYWORDS_SET_A)}
</section>
<section>
<p><strong>FM term set B</strong></p>
{keyword_list_html(FM_KEYWORDS_SET_B)}
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


def render_month_page(
    month: str,
    summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    digest: dict[str, Any],
) -> str:
    del summaries, metadata, digest
    from .site_payload import month_label

    month_attr = html.escape(month)
    month_json = html.escape(f"../../digest/{month}/papers.json")
    manifest_json = html.escape("../../data/months.json")
    month_title = html.escape(month_label(month))
    month_tab_title = html.escape(tab_title(month_tab_label(month)))
    nav = nav_html("../../index.html", "../../explore/index.html", "../../process/index.html", "home")
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{month_tab_title}</title>
<link rel='stylesheet' href='../../assets/style.css?v={ASSET_VERSION}'></head>
<body>
  {nav}
  <main id='digest-app' class='container' data-view='month' data-month='{month_attr}' data-manifest-json='{manifest_json}' data-month-json='{month_json}'>
    <section class='hero-banner month-hero'>
      <p class='hero-kicker'>Monthly Digest</p>
      <h1>{month_title} Digest</h1>
      <p class='sub'>Accepted EEG-FM papers and summaries for this month.</p>
    </section>
    <section id='featured-paper'></section>
    <section id='controls' class='controls'></section>
    <p id='results-meta' class='small'></p>
    <section id='results'></section>
  </main>
  <script src='../../assets/site.js?v={ASSET_VERSION}'></script>
</body></html>
"""


def render_home_page(months: list[str]) -> str:
    fallback_months = html.escape(json.dumps(months, ensure_ascii=False))
    nav = nav_html("index.html", "explore/index.html", "process/index.html", "home")
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(tab_title())}</title>
<link rel='stylesheet' href='assets/style.css?v={ASSET_VERSION}'></head><body>
{nav}
<main id='digest-app' class='container' data-view='home' data-month='' data-manifest-json='data/months.json' data-fallback-months='{fallback_months}'>
{about_digest_block("process/index.html", include_process_cta=False)}
<section id='home-controls' class='controls'></section>
<section id='home-results'></section>
</main>
<script src='assets/site.js?v={ASSET_VERSION}'></script>
</body></html>
"""


def render_explore_page(months: list[str]) -> str:
    fallback_months = html.escape(json.dumps(months, ensure_ascii=False))
    nav = nav_html("../index.html", "../explore/index.html", "../process/index.html", "explore")
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(tab_title("Search"))}</title>
<link rel='stylesheet' href='../assets/style.css?v={ASSET_VERSION}'></head><body>
{nav}
<main id='digest-app' class='container' data-view='explore' data-month='' data-manifest-json='../data/months.json' data-fallback-months='{fallback_months}'>
<h1>Search</h1>
<section id='controls' class='controls'></section>
<p id='results-meta' class='small'></p>
<section id='results'></section>
</main>
<script src='../assets/site.js?v={ASSET_VERSION}'></script>
</body></html>
"""
