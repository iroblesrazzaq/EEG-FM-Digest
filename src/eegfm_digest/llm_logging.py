"""Structured stderr logging for LLM stage failures (no secrets or full text)."""

from __future__ import annotations

import sys
from typing import Any


def log_stage_failure(
    stage: str,
    *,
    arxiv_id_base: str,
    provider: str,
    model: str,
    exc: BaseException,
    repair_used: bool | None = None,
    attempt: int | None = None,
) -> None:
    status_code = getattr(exc, "status_code", None)
    parts = [
        f"[{stage}]",
        f"arxiv_id={arxiv_id_base}",
        f"provider={provider}",
        f"model={model}",
        f"error={type(exc).__name__}",
    ]
    if status_code is not None:
        parts.append(f"status={status_code}")
    if attempt is not None:
        parts.append(f"attempt={attempt}")
    if repair_used is not None:
        parts.append(f"repair_used={repair_used}")
    detail = str(exc).strip()
    if detail:
        parts.append(f"detail={detail[:220]}")
    print(" ".join(parts), file=sys.stderr)


def log_summary_attempt(arxiv_id_base: str, attempt: dict[str, Any] | None) -> None:
    if not isinstance(attempt, dict):
        print(f"[summary_attempt] arxiv_id={arxiv_id_base} category=unknown", file=sys.stderr)
        return
    category = str(attempt.get("category") or "unknown")
    error = str(attempt.get("error") or "").strip()
    parts = [f"[summary_attempt] arxiv_id={arxiv_id_base}", f"category={category}"]
    if error:
        parts.append(f"error={error[:220]}")
    print(" ".join(parts), file=sys.stderr)


def log_daily_failure_summary(
    *,
    triage_failures: int,
    summary_failures: int,
    failed_triage_ids: list[str] | None = None,
    failed_summary_ids: list[str] | None = None,
    summary_categories: list[str] | None = None,
) -> None:
    if triage_failures == 0 and summary_failures == 0:
        return
    parts = [
        "[daily] LLM partial failures:",
        f"triage={triage_failures}",
        f"summary={summary_failures}",
    ]
    if failed_triage_ids:
        parts.append(f"failed_triage_ids={','.join(failed_triage_ids[:20])}")
    if failed_summary_ids:
        parts.append(f"failed_summary_ids={','.join(failed_summary_ids[:20])}")
    if summary_categories:
        parts.append(f"summary_categories={','.join(summary_categories[:20])}")
    print(" ".join(parts), file=sys.stderr)
