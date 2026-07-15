from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eegfm_digest.arxiv import fetch_paper_by_id, parse_arxiv_id
from eegfm_digest.config import load_config
from eegfm_digest.llm import LLMCallConfig, build_llm_call, load_api_key
from eegfm_digest.triage import load_schema, triage_paper


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _find_existing_review(output_dir: Path, arxiv_id_base: str) -> tuple[dict[str, Any], Path] | None:
    for backend_path in sorted(output_dir.glob("*/backend_rows.jsonl"), reverse=True):
        for row in _iter_jsonl(backend_path):
            if row.get("arxiv_id_base") == arxiv_id_base and isinstance(row.get("triage"), dict):
                return row, backend_path
    return None


def _format_markdown(
    paper: dict[str, Any],
    triage: dict[str, Any],
    *,
    existing_path: Path | None = None,
) -> str:
    authors = paper.get("authors") or []
    reasons = triage.get("reasons") or []
    if not isinstance(authors, list):
        authors = [str(authors)]
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    abs_url = str(paper.get("links", {}).get("abs") or f"https://arxiv.org/abs/{paper['arxiv_id_base']}")
    lines = [
        f"# {paper.get('title', paper['arxiv_id_base'])}",
        "",
        f"- **Authors:** {', '.join(str(author) for author in authors) or 'Unknown'}",
        f"- **Decision:** `{triage.get('decision', 'reject')}`",
        f"- **Confidence:** {float(triage.get('confidence', 0.0)):.2f}",
        f"- **arXiv:** [{paper['arxiv_id_base']}]({abs_url})",
    ]
    if existing_path is not None:
        lines.append(f"- **Reviewed In:** `{existing_path.as_posix()}`")
    lines.extend(["", "## Reasons", ""])
    for reason in reasons:
        lines.append(f"- {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EEG-FM triage for a single arXiv paper.")
    parser.add_argument("arxiv_id", help="arXiv identifier, e.g. 2401.12345")
    args = parser.parse_args(argv)

    arxiv_id_base, _version = parse_arxiv_id(args.arxiv_id)
    paper = fetch_paper_by_id(arxiv_id_base)
    if paper is None:
        print(f"No arXiv paper found for `{arxiv_id_base}`.", file=sys.stderr)
        return 1

    cfg = load_config()
    existing = _find_existing_review(cfg.output_dir, paper["arxiv_id_base"])
    if existing is not None:
        row, backend_path = existing
        print(_format_markdown(row, row["triage"], existing_path=backend_path))
        return 0

    triage_schema = load_schema(Path("schemas/triage.json"))
    triage_prompt = _read("prompts/triage.md")
    repair_prompt = _read("prompts/repair_json.md")
    llm_config = LLMCallConfig(
        provider="openrouter",
        api_key=load_api_key(),
        model=cfg.llm_model_triage,
        temperature=cfg.llm_temperature_triage,
        max_output_tokens=cfg.llm_max_output_tokens_triage,
        base_url="https://openrouter.ai/api/v1",
    )
    llm = build_llm_call(llm_config)
    try:
        triage = triage_paper(
            paper=paper,
            llm=llm,
            prompt_template=triage_prompt,
            repair_template=repair_prompt,
            schema=triage_schema,
        )
    finally:
        llm.close()

    print(_format_markdown(paper, triage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
