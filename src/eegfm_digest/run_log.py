"""Persistent record of the last successful daily run.

Stored as git-tracked JSON at ``data/last_successful_run.json`` so that
commit history doubles as an audit log.  The file is written only after
a run completes within the success criteria (no arXiv 5xx / no LLM
quota errors); on failure it is left untouched so the next run's
window auto-extends.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

CURRENT_VERSION = 1


class RunLogError(Exception):
    """Raised when the run log file is malformed or unreadable."""


@dataclass(frozen=True)
class RunLog:
    """Single-row state describing the most recent successful run."""

    last_success_utc: str
    last_query_end_utc: str
    papers_fetched: int = 0
    papers_accepted: int = 0
    affected_months: tuple[str, ...] = ()
    run_id: str = ""
    version: int = CURRENT_VERSION

    def to_dict(self) -> dict:
        return {
            "affected_months": list(self.affected_months),
            "last_query_end_utc": self.last_query_end_utc,
            "last_success_utc": self.last_success_utc,
            "papers_accepted": self.papers_accepted,
            "papers_fetched": self.papers_fetched,
            "run_id": self.run_id,
            "version": self.version,
        }


def format_utc(dt: datetime) -> str:
    """Render a datetime as ``YYYY-MM-DDTHH:MM:SSZ`` in UTC (second precision)."""
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 string; accepts trailing ``Z`` or explicit offset."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_run_log(path: Path) -> RunLog | None:
    """Return the stored run log, or ``None`` if the file does not exist."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunLogError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunLogError(f"run log at {path} is not a JSON object")

    version = int(data.get("version", 1))
    if version != CURRENT_VERSION:
        raise RunLogError(
            f"unsupported run log version {version} (expected {CURRENT_VERSION})"
        )

    try:
        last_query_end_utc = str(data["last_query_end_utc"])
        parse_utc(last_query_end_utc)
    except (KeyError, ValueError) as exc:
        raise RunLogError(f"invalid last_query_end_utc in {path}: {exc}") from exc

    affected = data.get("affected_months") or []
    return RunLog(
        last_success_utc=str(data.get("last_success_utc", last_query_end_utc)),
        last_query_end_utc=last_query_end_utc,
        papers_fetched=int(data.get("papers_fetched", 0)),
        papers_accepted=int(data.get("papers_accepted", 0)),
        affected_months=tuple(str(m) for m in affected),
        run_id=str(data.get("run_id", "")),
        version=version,
    )


def save_run_log(path: Path, log: RunLog) -> None:
    """Write the run log to disk (creating parent dirs as needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = log.to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compute_since(
    log: RunLog | None,
    overlap_hours: float,
    default_lookback_hours: float = 24.0,
    now: datetime | None = None,
) -> tuple[datetime, bool]:
    """Derive the ``since`` timestamp for the next daily run.

    Returns ``(since_utc, used_default)``.  When a run log exists the window
    starts at ``last_query_end_utc - overlap_hours``; otherwise it starts at
    ``now - default_lookback_hours`` (first-ever run fallback).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)

    if log is None:
        return now - timedelta(hours=default_lookback_hours), True

    end = parse_utc(log.last_query_end_utc)
    return end - timedelta(hours=overlap_hours), False
