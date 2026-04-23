from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from dateutil.relativedelta import relativedelta

from .config import load_config
from .pipeline import run_month, run_window
from .run_log import RunLog, compute_since, format_utc, load_run_log, save_run_log

RUN_LOG_FILENAME = "last_successful_run.json"

DEFAULT_OVERLAP_HOURS = 6.0
DEFAULT_LOOKBACK_HOURS_FIRST_RUN = 24.0


def default_month() -> str:
    first = date.today().replace(day=1)
    prev = first - relativedelta(months=1)
    return f"{prev.year:04d}-{prev.month:02d}"


def _parse_iso(raw: str) -> datetime:
    """Parse an ISO-8601 CLI argument, treating naive inputs as UTC."""
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid ISO-8601 datetime {raw!r}; expected e.g. 2026-04-22T10:00:00Z"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_daily_window(
    run_log_path: Path,
    overlap_hours: float,
    now: datetime,
    cli_since: datetime | None,
    cli_until: datetime | None,
) -> tuple[datetime, datetime, RunLog | None, bool]:
    log = load_run_log(run_log_path)
    until = cli_until or now

    if cli_since is not None:
        return cli_since, until, log, False

    since, used_default = compute_since(
        log,
        overlap_hours=overlap_hours,
        default_lookback_hours=DEFAULT_LOOKBACK_HOURS_FIRST_RUN,
        now=now,
    )
    return since, until, log, used_default


def _run_daily(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.max_candidates is not None:
        cfg = replace(cfg, max_candidates=args.max_candidates)
    if args.max_accepted is not None:
        cfg = replace(cfg, max_accepted=args.max_accepted)
    if args.include_borderline:
        cfg = replace(cfg, include_borderline=True)

    run_log_path = cfg.data_dir / RUN_LOG_FILENAME
    now = datetime.now(timezone.utc)

    since, until, prior_log, used_default = _resolve_daily_window(
        run_log_path,
        overlap_hours=args.overlap_hours,
        now=now,
        cli_since=args.since,
        cli_until=args.until,
    )

    if used_default:
        print(
            f"[daily] WARNING: no prior run log at {run_log_path}; "
            f"falling back to {DEFAULT_LOOKBACK_HOURS_FIRST_RUN:.0f}h lookback. "
            f"Use --since to override for historical backfills.",
            file=sys.stderr,
        )

    print(
        f"[daily] window since={format_utc(since)} until={format_utc(until)} "
        f"overlap_hours={args.overlap_hours} prior_run_log={'yes' if prior_log else 'no'}"
    )

    stats = run_window(
        cfg,
        since,
        until,
        no_pdf=args.no_pdf,
        no_site=args.no_site,
        force=args.force,
    )

    print(
        f"[daily] fetched={stats.window_candidates} "
        f"affected_months={list(stats.affected_months)} "
        f"accepted={stats.total_accepted} "
        f"triage_failures={stats.total_triage_failures} "
        f"summary_failures={stats.total_summary_failures}"
    )
    for m in stats.per_month:
        print(
            f"[daily]   month={m.month} candidates={m.candidates} "
            f"accepted={m.accepted} summarized={m.summarized} "
            f"triage_failures={m.triage_failures} summary_failures={m.summary_failures}"
        )

    new_log = RunLog(
        last_success_utc=format_utc(now),
        last_query_end_utc=format_utc(until),
        papers_fetched=stats.window_candidates,
        papers_accepted=stats.total_accepted,
        affected_months=stats.affected_months,
        run_id=format_utc(now),
    )
    partial_failures = stats.total_triage_failures + stats.total_summary_failures
    if partial_failures:
        print(
            f"[daily] WARNING: {partial_failures} paper(s) failed LLM processing "
            f"(triage={stats.total_triage_failures} summary={stats.total_summary_failures}). "
            "Run log NOT advanced; failed papers will be retried next run.",
            file=sys.stderr,
        )
        return 1

    if not args.dry_run:
        save_run_log(run_log_path, new_log)
        print(f"[daily] success=true wrote {run_log_path}")
    else:
        print("[daily] success=true (dry-run: run log not written)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default=None, help="YYYY-MM (default: previous month)")
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Run in daily mode: query since last successful run (minus overlap) to --until/now",
    )
    parser.add_argument(
        "--since",
        type=_parse_iso,
        default=None,
        help="ISO-8601 start of the window (daily mode; overrides run log)",
    )
    parser.add_argument(
        "--until",
        type=_parse_iso,
        default=None,
        help="ISO-8601 end of the window (daily mode; defaults to current time)",
    )
    parser.add_argument(
        "--overlap-hours",
        type=float,
        default=DEFAULT_OVERLAP_HOURS,
        help=f"Back-overlap in hours to absorb arXiv indexing lag (default {DEFAULT_OVERLAP_HOURS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Daily mode only: skip writing the run log (useful for verification runs)",
    )
    parser.add_argument("--feature-paper", default=None, help="Accepted arXiv id to feature for this month")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-accepted", type=int, default=None)
    parser.add_argument("--include-borderline", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--no-site", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.daily:
        if args.month is not None:
            parser.error("--daily and --month are mutually exclusive")
        if args.feature_paper is not None:
            parser.error("--feature-paper is not supported in --daily mode")
        sys.exit(_run_daily(args))

    daily_only_set = [
        flag
        for flag, was_set in (
            ("--since", args.since is not None),
            ("--until", args.until is not None),
            ("--dry-run", args.dry_run),
        )
        if was_set
    ]
    if daily_only_set:
        parser.error(
            f"{', '.join(daily_only_set)} require --daily (month mode ignores them)"
        )

    month = args.month or default_month()
    cfg = load_config()
    if args.max_candidates is not None:
        cfg = replace(cfg, max_candidates=args.max_candidates)
    if args.max_accepted is not None:
        cfg = replace(cfg, max_accepted=args.max_accepted)
    if args.include_borderline:
        cfg = replace(cfg, include_borderline=True)

    run_month(
        cfg,
        month,
        no_pdf=args.no_pdf,
        no_site=args.no_site,
        force=args.force,
        feature_paper=args.feature_paper,
    )


if __name__ == "__main__":
    main()
