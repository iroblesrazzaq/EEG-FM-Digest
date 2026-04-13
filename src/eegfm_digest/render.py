from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def pick_top_picks(summaries: list[dict[str, Any]], triage_map: dict[str, dict[str, Any]]) -> list[str]:
    ranked = sorted(
        summaries,
        key=lambda s: (
            float(triage_map.get(s["arxiv_id_base"], {}).get("confidence", 0.0)),
            len([k for k in s.get("key_points", []) if k and k != "unknown"]),
            s["arxiv_id_base"],
        ),
        reverse=True,
    )
    return [s["arxiv_id_base"] for s in ranked[:5]]


def _normalize_featured_paper(
    featured_paper: str | None,
    triage_rows: list[dict[str, Any]],
) -> str | None:
    if featured_paper is None:
        return None
    featured_paper_id = str(featured_paper).strip()
    if not featured_paper_id:
        return None
    accepted_ids = {
        str(row.get("arxiv_id_base", "")).strip()
        for row in triage_rows
        if str(row.get("decision", "")).strip() == "accept"
    }
    if featured_paper_id not in accepted_ids:
        raise RuntimeError(
            f"Featured paper {featured_paper_id} was not accepted for this month; only accepted papers can be featured."
        )
    return featured_paper_id


def build_digest(
    month: str,
    candidates: list[dict[str, Any]],
    triage_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    featured_paper: str | None = None,
) -> dict[str, Any]:
    triage_map = {t["arxiv_id_base"]: t for t in triage_rows}
    sections_map: dict[str, list[str]] = defaultdict(list)
    for s in sorted(summaries, key=lambda x: (x["paper_type"], x["published_date"], x["arxiv_id_base"])):
        sections_map[s["paper_type"]].append(s["arxiv_id_base"])

    return {
        "month": month,
        "stats": {
            "candidates": len(candidates),
            "accepted": len([t for t in triage_rows if t["decision"] == "accept"]),
            "summarized": len(summaries),
        },
        "featured_paper": _normalize_featured_paper(featured_paper, triage_rows),
        "top_picks": pick_top_picks(summaries, triage_map),
        "sections": [{"title": k, "paper_ids": v} for k, v in sorted(sections_map.items())],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
