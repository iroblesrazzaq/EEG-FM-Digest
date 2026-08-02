"""Characterization tests for daily-mode LLM failure handling."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from eegfm_digest.config import Config
from eegfm_digest.llm import LLMRateLimitError
from eegfm_digest.pipeline import run_month
from eegfm_digest.run import _run_daily
from eegfm_digest.run_log import load_run_log


def _candidate(arxiv_id_base: str, published: str, title: str = "Paper") -> dict:
    return {
        "arxiv_id": f"{arxiv_id_base}v1",
        "arxiv_id_base": arxiv_id_base,
        "version": 1,
        "title": title,
        "summary": f"{title} abstract",
        "authors": ["Author A"],
        "categories": ["cs.LG"],
        "published": published,
        "updated": published,
        "links": {"abs": f"https://arxiv.org/abs/{arxiv_id_base}", "pdf": f"https://arxiv.org/pdf/{arxiv_id_base}.pdf"},
    }


def _cfg(tmp_path: Path) -> Config:
    return Config(
        llm_model_triage="triage-model",
        llm_model_summary="summary-model",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
        arxiv_rate_limit_seconds=0.0,
        pdf_rate_limit_seconds=0.0,
    )


def _stub_llm(monkeypatch):
    class DummyLMCall:
        def close(self):
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_a, **_k: "test-key")
    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_a, **_k: DummyLMCall())


def _stub_pdf(monkeypatch):
    def fake_download_pdf(_url, out_path, _rate):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4")
        return out_path

    def fake_extract_text(_pdf_path, text_path):
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("Abstract\nEEG\n\nMethods\nM", encoding="utf-8")
        return {"tool": "pypdf", "pages": 1, "chars": 20, "error": None}

    monkeypatch.setattr("eegfm_digest.pipeline.download_pdf", fake_download_pdf)
    monkeypatch.setattr("eegfm_digest.pipeline.extract_text", fake_extract_text)


def test_pipeline_triage_failure_skips_db_and_increments_stats(monkeypatch, tmp_path: Path):
    candidate = _candidate("2501.00001", "2025-01-02T00:00:00Z")
    monkeypatch.setattr(
        "eegfm_digest.pipeline.fetch_month_candidates",
        lambda *_a, **_k: [candidate],
    )
    _stub_llm(monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("triage exploded")

    monkeypatch.setattr("eegfm_digest.pipeline.triage_paper", boom)

    cfg = _cfg(tmp_path)
    stats = run_month(cfg, "2025-01", no_site=True, no_pdf=True)

    assert stats.triage_failures == 1
    assert stats.failed_triage_ids == ("2501.00001",)
    assert stats.summarized == 0

    conn = sqlite3.connect(cfg.data_dir / "digest.sqlite")
    try:
        row = conn.execute(
            "SELECT 1 FROM triage WHERE arxiv_id_base = ?",
            ("2501.00001",),
        ).fetchone()
    finally:
        conn.close()
    assert row is None


def test_pipeline_summary_failure_skips_db_and_increments_stats(monkeypatch, tmp_path: Path):
    candidate = _candidate("2501.00001", "2025-01-02T00:00:00Z")
    monkeypatch.setattr(
        "eegfm_digest.pipeline.fetch_month_candidates",
        lambda *_a, **_k: [candidate],
    )
    _stub_llm(monkeypatch)
    _stub_pdf(monkeypatch)

    monkeypatch.setattr(
        "eegfm_digest.pipeline.triage_paper",
        lambda paper, *_a, **_k: {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["ok"],
        },
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("summary exploded")

    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper", boom)

    cfg = _cfg(tmp_path)
    stats = run_month(cfg, "2025-01", no_site=True)

    assert stats.summary_failures == 1
    assert stats.failed_summary_ids == ("2501.00001",)
    assert stats.summarized == 0

    conn = sqlite3.connect(cfg.data_dir / "digest.sqlite")
    try:
        row = conn.execute(
            "SELECT 1 FROM summaries WHERE arxiv_id_base = ?",
            ("2501.00001",),
        ).fetchone()
    finally:
        conn.close()
    assert row is None


def test_pipeline_rate_limit_propagates(monkeypatch, tmp_path: Path):
    candidate = _candidate("2501.00001", "2025-01-02T00:00:00Z")
    monkeypatch.setattr(
        "eegfm_digest.pipeline.fetch_month_candidates",
        lambda *_a, **_k: [candidate],
    )
    _stub_llm(monkeypatch)

    def rate_limited(*_args, **_kwargs):
        raise LLMRateLimitError("quota exhausted")

    monkeypatch.setattr("eegfm_digest.pipeline.triage_paper", rate_limited)

    with pytest.raises(LLMRateLimitError):
        run_month(_cfg(tmp_path), "2025-01", no_site=True, no_pdf=True)


def test_daily_mode_returns_nonzero_without_advancing_run_log(monkeypatch, tmp_path: Path, capsys):
    candidate = _candidate("2501.00001", "2025-01-02T00:00:00Z")
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("eegfm_digest.run.load_config", lambda: cfg)

    class Args:
        max_candidates = None
        max_accepted = None
        include_borderline = False
        overlap_hours = 6.0
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        until = datetime(2026, 1, 2, tzinfo=timezone.utc)
        no_pdf = True
        no_site = True
        force = False
        dry_run = False

    window_stats = type(
        "WindowRunStats",
        (),
        {
            "window_candidates": 1,
            "affected_months": ("2025-01",),
            "total_accepted": 0,
            "total_triage_failures": 1,
            "total_summary_failures": 0,
            "failed_triage_ids": ("2501.00001",),
            "failed_summary_ids": (),
            "per_month": (
                type(
                    "MonthRunStats",
                    (),
                    {
                        "month": "2025-01",
                        "candidates": 1,
                        "accepted": 0,
                        "summarized": 0,
                        "triage_failures": 1,
                        "summary_failures": 0,
                    },
                )(),
            ),
        },
    )()

    monkeypatch.setattr("eegfm_digest.run.run_window", lambda *_a, **_k: window_stats)

    log_path = cfg.data_dir / "last_successful_run.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "last_success_utc": "2026-01-01T00:00:00Z",
                "last_query_end_utc": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    exit_code = _run_daily(Args())
    assert exit_code == 1
    assert load_run_log(log_path) is not None
    assert "last_query_end_utc" in log_path.read_text(encoding="utf-8")
    assert "2026-01-01T00:00:00Z" in log_path.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "Run log NOT advanced" in captured.err
