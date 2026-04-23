import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from eegfm_digest.run_log import (
    CURRENT_VERSION,
    RunLog,
    RunLogError,
    compute_since,
    format_utc,
    load_run_log,
    parse_utc,
    save_run_log,
)


def test_format_and_parse_roundtrip():
    dt = datetime(2026, 4, 22, 14, 37, 12, 500_000, tzinfo=timezone.utc)
    formatted = format_utc(dt)
    assert formatted == "2026-04-22T14:37:12Z"
    parsed = parse_utc(formatted)
    assert parsed == dt.replace(microsecond=0)


def test_parse_accepts_explicit_offset():
    assert parse_utc("2026-04-22T10:00:00+00:00") == datetime(
        2026, 4, 22, 10, 0, tzinfo=timezone.utc
    )


def test_parse_naive_treated_as_utc():
    # Naive input is treated as UTC by convention
    assert parse_utc("2026-04-22T10:00:00") == datetime(
        2026, 4, 22, 10, 0, tzinfo=timezone.utc
    )


def test_load_missing_returns_none(tmp_path: Path):
    assert load_run_log(tmp_path / "nope.json") is None


def test_save_then_load_roundtrip(tmp_path: Path):
    path = tmp_path / "run.json"
    log = RunLog(
        last_success_utc="2026-04-22T14:37:00Z",
        last_query_end_utc="2026-04-22T14:37:00Z",
        papers_fetched=12,
        papers_accepted=3,
        affected_months=("2026-04",),
        run_id="manual-2026-04-22",
    )
    save_run_log(path, log)
    round_tripped = load_run_log(path)
    assert round_tripped == log


def test_save_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "run.json"
    save_run_log(
        path,
        RunLog(last_success_utc="2026-04-22T10:00:00Z", last_query_end_utc="2026-04-22T10:00:00Z"),
    )
    assert path.exists()


def test_save_is_atomic_no_leftover_tmp(tmp_path: Path):
    path = tmp_path / "run.json"
    save_run_log(
        path,
        RunLog(last_success_utc="2026-04-22T10:00:00Z", last_query_end_utc="2026-04-22T10:00:00Z"),
    )
    # The atomic write uses <name>.tmp as a staging file; it must be gone
    # after a successful replace.
    assert not (tmp_path / "run.json.tmp").exists()


def test_save_preserves_prior_file_when_replace_fails(tmp_path: Path, monkeypatch):
    """If os.replace raises, the pre-existing file must remain intact."""
    import eegfm_digest.run_log as run_log_mod

    path = tmp_path / "run.json"
    original = RunLog(
        last_success_utc="2026-04-22T10:00:00Z",
        last_query_end_utc="2026-04-22T10:00:00Z",
        run_id="original",
    )
    save_run_log(path, original)
    original_bytes = path.read_bytes()

    def boom(src, dst):  # noqa: ARG001
        raise OSError("simulated crash after staging")

    monkeypatch.setattr(run_log_mod.os, "replace", boom)
    replacement = RunLog(
        last_success_utc="2026-04-23T10:00:00Z",
        last_query_end_utc="2026-04-23T10:00:00Z",
        run_id="replacement",
    )
    with pytest.raises(OSError, match="simulated crash"):
        save_run_log(path, replacement)

    # Target file is untouched.
    assert path.read_bytes() == original_bytes


def test_save_is_deterministic(tmp_path: Path):
    """Stable key order + trailing newline so git diffs stay minimal."""
    path = tmp_path / "run.json"
    log = RunLog(last_success_utc="2026-04-22T10:00:00Z", last_query_end_utc="2026-04-22T10:00:00Z")
    save_run_log(path, log)
    first = path.read_text(encoding="utf-8")
    save_run_log(path, log)
    second = path.read_text(encoding="utf-8")
    assert first == second
    assert first.endswith("\n")
    # sort_keys -> affected_months appears before last_query_end_utc
    assert first.index('"affected_months"') < first.index('"last_query_end_utc"')


def test_load_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "run.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RunLogError, match="invalid JSON"):
        load_run_log(path)


def test_load_rejects_non_object(tmp_path: Path):
    path = tmp_path / "run.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RunLogError, match="not a JSON object"):
        load_run_log(path)


def test_load_rejects_unknown_version(tmp_path: Path):
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "last_query_end_utc": "2026-04-22T10:00:00Z",
                "last_success_utc": "2026-04-22T10:00:00Z",
                "version": 999,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RunLogError, match="unsupported run log version"):
        load_run_log(path)


def test_load_rejects_missing_timestamp(tmp_path: Path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"version": CURRENT_VERSION}), encoding="utf-8")
    with pytest.raises(RunLogError, match="invalid last_query_end_utc"):
        load_run_log(path)


def test_compute_since_uses_overlap_when_log_exists():
    log = RunLog(
        last_success_utc="2026-04-22T10:00:00Z",
        last_query_end_utc="2026-04-22T10:00:00Z",
    )
    now = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    since, used_default = compute_since(log, overlap_hours=6, now=now)
    assert used_default is False
    assert since == datetime(2026, 4, 22, 4, 0, tzinfo=timezone.utc)


def test_compute_since_falls_back_when_no_log():
    now = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    since, used_default = compute_since(
        None, overlap_hours=6, default_lookback_hours=24, now=now
    )
    assert used_default is True
    assert since == now - timedelta(hours=24)


def test_compute_since_zero_overlap():
    log = RunLog(
        last_success_utc="2026-04-22T10:00:00Z",
        last_query_end_utc="2026-04-22T10:00:00Z",
    )
    now = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    since, _ = compute_since(log, overlap_hours=0, now=now)
    assert since == datetime(2026, 4, 22, 10, 0, tzinfo=timezone.utc)
