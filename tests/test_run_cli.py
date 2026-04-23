import argparse
from datetime import datetime, timezone
from pathlib import Path

import pytest

from eegfm_digest.run import _parse_iso, _resolve_daily_window
from eegfm_digest.run_log import RunLog, save_run_log


def test_parse_iso_accepts_z_suffix():
    assert _parse_iso("2026-04-22T10:00:00Z") == datetime(
        2026, 4, 22, 10, 0, tzinfo=timezone.utc
    )


def test_parse_iso_accepts_offset():
    assert _parse_iso("2026-04-22T06:00:00-04:00") == datetime(
        2026, 4, 22, 10, 0, tzinfo=timezone.utc
    )


def test_parse_iso_treats_naive_as_utc():
    assert _parse_iso("2026-04-22T10:00:00") == datetime(
        2026, 4, 22, 10, 0, tzinfo=timezone.utc
    )


def test_parse_iso_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError, match="invalid ISO-8601"):
        _parse_iso("not a date")


def test_resolve_window_uses_prior_log_minus_overlap(tmp_path: Path):
    log_path = tmp_path / "last_successful_run.json"
    save_run_log(
        log_path,
        RunLog(
            last_success_utc="2026-04-22T10:00:00Z",
            last_query_end_utc="2026-04-22T10:00:00Z",
        ),
    )
    now = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)

    since, until, log, used_default = _resolve_daily_window(
        log_path,
        overlap_hours=6,
        now=now,
        cli_since=None,
        cli_until=None,
    )
    assert used_default is False
    assert log is not None
    assert since == datetime(2026, 4, 22, 4, 0, tzinfo=timezone.utc)
    assert until == now


def test_resolve_window_falls_back_when_no_log(tmp_path: Path):
    log_path = tmp_path / "missing.json"
    now = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)

    since, until, log, used_default = _resolve_daily_window(
        log_path,
        overlap_hours=6,
        now=now,
        cli_since=None,
        cli_until=None,
    )
    assert used_default is True
    assert log is None
    assert since == datetime(2026, 4, 22, 10, 0, tzinfo=timezone.utc)
    assert until == now


def test_resolve_window_cli_since_overrides_log(tmp_path: Path):
    log_path = tmp_path / "last_successful_run.json"
    save_run_log(
        log_path,
        RunLog(
            last_success_utc="2026-04-22T10:00:00Z",
            last_query_end_utc="2026-04-22T10:00:00Z",
        ),
    )
    now = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    override_since = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)

    since, until, log, used_default = _resolve_daily_window(
        log_path,
        overlap_hours=6,
        now=now,
        cli_since=override_since,
        cli_until=None,
    )
    assert used_default is False
    assert since == override_since
    assert until == now
    assert log is not None  # still loaded


def test_resolve_window_cli_until_overrides_now(tmp_path: Path):
    log_path = tmp_path / "missing.json"
    now = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    override_until = datetime(2026, 4, 22, 23, 0, tzinfo=timezone.utc)

    _, until, _, _ = _resolve_daily_window(
        log_path,
        overlap_hours=6,
        now=now,
        cli_since=None,
        cli_until=override_until,
    )
    assert until == override_until
