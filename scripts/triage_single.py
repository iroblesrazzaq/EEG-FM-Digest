#!/usr/bin/env python3
"""Triage a single arXiv paper and optionally seed it as a digest straggler.

On accept, upserts ``papers`` + ``triage`` into SQLite without summarizing.
The next daily ``run_window`` stragglers sweep picks up PDF+summary work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eegfm_digest.arxiv import fetch_paper_by_id, parse_arxiv_id
from eegfm_digest.cache_meta import build_stage_metadata
from eegfm_digest.config import load_config
from eegfm_digest.db import DigestDB
from eegfm_digest.llm import LLMCallConfig, build_llm_call, load_api_key, provider_base_url
from eegfm_digest.stage_context import load_triage_stage_context
from eegfm_digest.triage import triage_paper


def _format_markdown(
    paper: dict[str, Any],
    triage: dict[str, Any],
    *,
    note: str | None = None,
) -> str:
    authors = paper.get("authors") or []
    reasons = triage.get("reasons") or []
    if not isinstance(authors, list):
        authors = [str(authors)]
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    abs_url = str(
        paper.get("links", {}).get("abs") or f"https://arxiv.org/abs/{paper['arxiv_id_base']}"
    )
    lines = [
        f"# {paper.get('title', paper['arxiv_id_base'])}",
        "",
        f"- **Authors:** {', '.join(str(author) for author in authors) or 'Unknown'}",
        f"- **Decision:** `{triage.get('decision', 'reject')}`",
        f"- **Confidence:** {float(triage.get('confidence', 0.0)):.2f}",
        f"- **arXiv:** [{paper['arxiv_id_base']}]({abs_url})",
    ]
    if note:
        lines.append(f"- **Note:** {note}")
    lines.extend(["", "## Reasons", ""])
    for reason in reasons:
        lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def _month_from_paper(paper: dict[str, Any]) -> str:
    published = str(paper.get("published", "")).strip()
    if len(published) >= 7:
        return published[:7]
    raise ValueError(f"paper {paper.get('arxiv_id_base')} missing published month")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Triage one arXiv paper; persist accepts as summary stragglers."
    )
    parser.add_argument("arxiv_id", help="arXiv identifier, e.g. 2401.12345")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run triage even if a triage row already exists in SQLite.",
    )
    args = parser.parse_args(argv)

    arxiv_id_base, _version = parse_arxiv_id(args.arxiv_id)
    paper = fetch_paper_by_id(arxiv_id_base)
    if paper is None:
        print(f"No arXiv paper found for `{arxiv_id_base}`.", file=sys.stderr)
        return 1

    cfg = load_config()
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    try:
        existing = None if args.force else db.get_triage(paper["arxiv_id_base"])
        if existing is not None:
            print(
                _format_markdown(
                    paper,
                    existing,
                    note="Already triaged in SQLite; skipped LLM call. Use --force to re-run.",
                )
            )
            return 0

        api_key = load_api_key(cfg.llm_provider)
        llm_config = LLMCallConfig(
            provider=cfg.llm_provider,
            api_key=api_key,
            model=cfg.llm_model_triage,
            temperature=cfg.llm_temperature_triage,
            max_output_tokens=cfg.llm_max_output_tokens_triage,
            base_url=provider_base_url(cfg.llm_provider),
        )
        triage_ctx = load_triage_stage_context(llm_config)
        llm = build_llm_call(llm_config)
        try:
            triage = triage_paper(
                paper=paper,
                llm=llm,
                prompt_template=triage_ctx.triage_prompt,
                repair_template=triage_ctx.repair_prompt,
                schema=triage_ctx.schema,
            )
        finally:
            llm.close()

        month = _month_from_paper(paper)
        db.upsert_paper(month, paper)
        db.upsert_triage(
            month,
            triage,
            meta=build_stage_metadata(
                triage_ctx.descriptor,
                repair_used=False,
                updated_at_source=str(paper.get("updated", "")).strip() or None,
            ),
        )

        decision = str(triage.get("decision", "reject"))
        if decision == "accept":
            note = (
                f"Persisted as straggler for month `{month}` "
                "(summary deferred to next daily run)."
            )
        else:
            note = f"Triage stored for month `{month}`; not queued for summary."

        print(_format_markdown(paper, triage, note=note))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
