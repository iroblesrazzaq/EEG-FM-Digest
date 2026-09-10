"""Public facade for static site generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .site_payload import month_manifest_item, month_payload
from .site_templates import (
    render_explore_page,
    render_home_page,
    render_models_page,
    render_month_page,
    render_process_page,
)

__all__ = [
    "refresh_html_shells",
    "render_explore_page",
    "render_home_page",
    "render_models_page",
    "render_month_page",
    "render_process_page",
    "update_home",
    "write_month_site",
]


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
    payload = month_payload(month, summaries, metadata, digest, backend_rows)
    (month_dir / "papers.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (month_dir / "digest.json").write_text(
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def update_home(docs_dir: Path) -> None:
    month_dirs = sorted(
        [p for p in (docs_dir / "digest").iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    ) if (docs_dir / "digest").exists() else []
    months = [p.name for p in month_dirs]
    (docs_dir / "index.html").write_text(render_home_page(months), encoding="utf-8")
    explore_dir = docs_dir / "explore"
    explore_dir.mkdir(parents=True, exist_ok=True)
    (explore_dir / "index.html").write_text(render_explore_page(months), encoding="utf-8")
    models_dir = docs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "index.html").write_text(render_models_page(), encoding="utf-8")
    process_dir = docs_dir / "process"
    process_dir.mkdir(parents=True, exist_ok=True)
    (process_dir / "index.html").write_text(render_process_page(), encoding="utf-8")
    manifest = {
        "latest": months[0] if months else None,
        "months": [month_manifest_item(month_dir) for month_dir in month_dirs],
    }
    data_dir = docs_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "months.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (docs_dir / ".nojekyll").write_text("\n", encoding="utf-8")


def refresh_html_shells(docs_dir: Path) -> None:
    """Rewrite static HTML shells (nav/assets) without touching papers.json."""
    digest_root = docs_dir / "digest"
    if digest_root.exists():
        for month_dir in sorted(p for p in digest_root.iterdir() if p.is_dir()):
            if not month_dir.name[0:4].isdigit():
                continue
            (month_dir / "index.html").write_text(
                render_month_page(month_dir.name, [], {}, {}),
                encoding="utf-8",
            )
    update_home(docs_dir)
