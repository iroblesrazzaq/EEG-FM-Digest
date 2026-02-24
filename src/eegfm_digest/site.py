from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


_TAG_LABELS: dict[str, dict[str, str]] = {
    "paper_type": {
        "new-model": "New Model",
        "eeg-fm": "New Model",
        "post-training": "Post-Training",
        "benchmark": "Benchmark",
        "survey": "Survey",
    },
    "backbone": {
        "transformer": "Transformer",
        "mamba-ssm": "Mamba-SSM",
        "moe": "MoE",
        "diffusion": "Diffusion",
    },
    "objective": {
        "masked-reconstruction": "Masked Reconstruction",
        "autoregressive": "Autoregressive",
        "contrastive": "Contrastive",
        "discrete-code-prediction": "Discrete Code Prediction",
    },
    "tokenization": {
        "time-patch": "Time Patch",
        "latent-tokens": "Latent Tokens",
        "discrete-tokens": "Discrete Tokens",
    },
    "topology": {
        "fixed-montage": "Fixed Montage",
        "channel-flexible": "Channel Flexible",
        "topology-agnostic": "Topology Agnostic",
    },
}


def _tag_value_label(category: str, value: str) -> str:
    mapping = _TAG_LABELS.get(category, {})
    if value in mapping:
        return mapping[value]
    return value.replace("-", " ").title()


def _tag_chips(summary: dict[str, Any]) -> str:
    tags = summary.get("tags")
    if not isinstance(tags, dict):
        return ""
    chips: list[str] = []
    for category in ("paper_type", "backbone", "objective", "tokenization", "topology"):
        values = tags.get(category, [])
        if not isinstance(values, list):
            continue
        for value in values:
            value_str = str(value).strip()
            chips.append(
                "<span "
                f"class='chip chip-{html.escape(category)}' "
                f"title='{html.escape(category.replace('_', ' '))}'>"
                f"{html.escape(_tag_value_label(category, value_str))}"
                "</span>"
            )
    if not chips:
        return ""
    return f"<p class='chips'>{' '.join(chips)}</p>"


def _summary_points(summary: dict[str, Any]) -> str:
    points = summary.get("key_points", [])
    if not isinstance(points, list):
        return ""
    bullets = [str(point).strip() for point in points if str(point).strip()][:3]
    if not bullets:
        return ""
    items = "".join(f"<li>{html.escape(point)}</li>" for point in bullets)
    return f"<ul class='summary-points'>{items}</ul>"


def _compact_detail(detail: str) -> str:
    # Light normalization prevents giant wall-of-text rendering.
    compact = re.sub(r"\s+", " ", detail).strip()
    return compact


def _card(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    title = html.escape(summary["title"])
    abs_url = html.escape(meta.get("links", {}).get("abs", "#"))
    authors = ", ".join(meta.get("authors", []))
    os_info = summary.get("open_source", {})
    code_link = (
        f"<a href='{html.escape(os_info['code_url'])}'>code</a>" if os_info.get("code_url") else ""
    )
    weights_link = (
        f"<a href='{html.escape(os_info['weights_url'])}'>weights</a>" if os_info.get("weights_url") else ""
    )
    detail = html.escape(_compact_detail(summary.get("detailed_summary", summary["one_liner"])))
    points_html = _summary_points(summary)
    tag_html = _tag_chips(summary)
    detail_html = (
        f"<details class='summary-detail'><summary>Detailed summary</summary><p>{detail}</p></details>"
        if detail
        else ""
    )
    return f"""
    <article class='paper-card' id='{html.escape(summary['arxiv_id_base'])}'>
      <h3><a href='{abs_url}'>{title}</a></h3>
      <div class='meta'>{html.escape(summary['published_date'])} · {html.escape(authors)}</div>
      <p><strong>Summary Highlights:</strong></p>
      {points_html}
      <p><strong>Unique contribution:</strong> {html.escape(summary['unique_contribution'])}</p>
      {detail_html}
      {tag_html}
      <p>{code_link} {weights_link}</p>
    </article>
    """


def _about_digest_block() -> str:
    return (
        "<section class='digest-about'>"
        "<h2>About This Digest</h2>"
        "<p>[why i made digest, how it works]</p>"
        "</section>"
    )


def render_month_page(month: str, summaries: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], digest: dict[str, Any]) -> str:
    by_id = {s["arxiv_id_base"]: s for s in summaries}
    sections_html: list[str] = []
    for section in digest.get("sections", []):
        cards = "\n".join(_card(by_id[pid], metadata.get(pid, {})) for pid in section["paper_ids"] if pid in by_id)
        sections_html.append(f"<section><h2>{html.escape(section['title'])}</h2>{cards}</section>")
    top_picks = ", ".join(digest.get("top_picks", []))
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>EEG-FM Digest {html.escape(month)}</title>
<link rel='stylesheet' href='../../assets/style.css'></head>
<body>
  <main>
    <h1>EEG Foundation Model Digest — {html.escape(month)}</h1>
    {_about_digest_block()}
    <p>Top picks: {html.escape(top_picks)}</p>
    {''.join(sections_html)}
  </main>
  <script src='../../assets/site.js'></script>
</body></html>
"""


def render_home_page(months: list[str]) -> str:
    latest = months[0] if months else ""
    links = "\n".join(
        f"<li><a href='digest/{m}/index.html'>{m}</a></li>" for m in months
    )
    latest_link = f"digest/{latest}/index.html" if latest else "#"
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>EEG-FM Digest</title>
<link rel='stylesheet' href='assets/style.css'></head><body>
<main>
<h1>EEG Foundation Model Digest</h1>
{_about_digest_block()}
<p>Latest month: <a href='{latest_link}'>{latest}</a></p>
<h2>Archive</h2>
<ul>{links}</ul>
</main>
<script src='assets/site.js'></script>
</body></html>
"""


def write_month_site(docs_dir: Path, month: str, summaries: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], digest: dict[str, Any]) -> None:
    month_dir = docs_dir / "digest" / month
    month_dir.mkdir(parents=True, exist_ok=True)
    (month_dir / "index.html").write_text(
        render_month_page(month, summaries, metadata, digest), encoding="utf-8"
    )
    (month_dir / "papers.json").write_text(
        json.dumps(sorted(summaries, key=lambda x: (x["published_date"], x["arxiv_id_base"])), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def update_home(docs_dir: Path) -> None:
    months = sorted(
        [p.name for p in (docs_dir / "digest").iterdir() if p.is_dir()],
        reverse=True,
    ) if (docs_dir / "digest").exists() else []
    (docs_dir / "index.html").write_text(render_home_page(months), encoding="utf-8")
    (docs_dir / ".nojekyll").write_text("\n", encoding="utf-8")
