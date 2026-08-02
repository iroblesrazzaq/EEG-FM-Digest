"""Paper selection for the summary stage."""

from __future__ import annotations

from typing import Any, Literal

BorderlinePolicy = Literal["pipeline", "batch"]


def select_papers_for_summary(
    candidates: list[dict[str, Any]],
    triage_map: dict[str, dict[str, Any]],
    *,
    include_borderline: bool,
    max_borderline_pdfs: int,
    max_accepted: int,
    borderline_policy: BorderlinePolicy = "pipeline",
) -> list[dict[str, Any]]:
    accepted = [
        p for p in candidates if triage_map.get(p["arxiv_id_base"], {}).get("decision") == "accept"
    ]
    if include_borderline:
        borderline = [
            p
            for p in candidates
            if triage_map.get(p["arxiv_id_base"], {}).get("decision") == "borderline"
        ]
        if borderline_policy == "pipeline":
            borderline = borderline[:max_borderline_pdfs]
        accepted.extend(borderline)
    return sorted(accepted, key=lambda x: (x["published"], x["arxiv_id_base"]))[:max_accepted]
