"""Helpers for the home-page monthly accepted-paper volume chart."""

from __future__ import annotations

from typing import Any


def accepted_counts_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return chronological ``{month, accepted, is_current}`` rows for charting."""
    latest = str(manifest.get("latest") or "").strip()
    months = manifest.get("months") or []
    if not isinstance(months, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in months:
        if not isinstance(item, dict):
            continue
        month = str(item.get("month") or "").strip()
        if not month:
            continue
        stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}
        try:
            accepted = int(stats.get("accepted", 0) or 0)
        except (TypeError, ValueError):
            accepted = 0
        rows.append(
            {
                "month": month,
                "accepted": max(0, accepted),
                "is_current": month == latest,
            }
        )
    rows.sort(key=lambda row: row["month"])
    return rows
