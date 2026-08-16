"""Tests for accept-without-summary (straggler) resummarization."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from eegfm_digest.config import Config
from eegfm_digest.db import DigestDB
from eegfm_digest.pipeline import (
    MonthRunStats,
    StragglerStats,
    _rerender_month_from_db,
    resummarize_stragglers,
    run_window,
)


def _candidate(arxiv_id_base: str, published: str, title: str) -> dict:
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
        "links": {
            "abs": f"https://arxiv.org/abs/{arxiv_id_base}",
            "pdf": f"https://arxiv.org/pdf/{arxiv_id_base}.pdf",
        },
    }


def _summary_payload(paper: dict) -> dict:
    return {
        "arxiv_id_base": paper["arxiv_id_base"],
        "title": paper["title"],
        "published_date": paper["published"][:10],
        "categories": paper["categories"],
        "paper_type": "method",
        "one_liner": "Concise summary line.",
        "detailed_summary": (
            "This work proposes a concise EEG modeling approach with explicit transfer framing "
            "and reports benchmark gains using pretrained representations."
        ),
        "unique_contribution": "Deterministic contribution sentence.",
        "key_points": ["point one", "point two", "point three"],
        "data_scale": {"datasets": [], "subjects": None, "eeg_hours": None, "channels": None},
        "method": {
            "architecture": "Transformer",
            "objective": None,
            "pretraining": None,
            "finetuning": None,
        },
        "evaluation": {"tasks": [], "benchmarks": [], "headline_results": []},
        "open_source": {"code_url": None, "weights_url": None, "license": None},
        "tags": {
            "paper_type": [],
            "backbone": [],
            "objective": [],
            "tokenization": [],
            "topology": [],
        },
        "limitations": [],
        "used_fulltext": True,
        "notes": "ok",
    }


def _cfg(tmp_path) -> Config:
    return Config(
        llm_provider="google",
        llm_model_triage="triage-model",
        llm_model_summary="summary-model",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
        max_candidates=20,
        max_accepted=20,
        arxiv_rate_limit_seconds=0.0,
        pdf_rate_limit_seconds=0.0,
    )


def _seed_accept_without_summary(db: DigestDB, paper: dict, month: str) -> None:
    db.upsert_paper(month, paper)
    db.upsert_triage(
        month,
        {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["r1", "r2"],
        },
    )


def _patch_summary_stack(monkeypatch, *, download_raises: bool = False) -> None:
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_a, **_k: "test-key")

    class DummyLMCall:
        def close(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_a, **_k: DummyLMCall())

    def fake_download_pdf(_url, out_path, _rate):  # noqa: ANN001
        if download_raises:
            raise RuntimeError("pdf boom")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4")
        return out_path

    def fake_extract_text(_pdf_path, text_path):  # noqa: ANN001
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(
            "Abstract\nEEG abstract\n\nIntroduction\nIntro\n\nMethods\nMethod\n\n"
            "Results\nResult\n\nConclusion\nEnd",
            encoding="utf-8",
        )
        return {"tool": "pypdf", "pages": 1, "chars": 100, "error": None}

    monkeypatch.setattr("eegfm_digest.pipeline.download_pdf", fake_download_pdf)
    monkeypatch.setattr("eegfm_digest.pipeline.extract_text", fake_extract_text)

    def fake_summarize(paper, *_a, **kwargs):  # noqa: ANN001
        payload = _summary_payload(paper)
        if "used_fulltext" in kwargs:
            payload["used_fulltext"] = kwargs["used_fulltext"]
        if "notes" in kwargs:
            payload["notes"] = kwargs["notes"]
        return payload

    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper", fake_summarize)


def test_get_accepted_without_summary(tmp_path):
    cfg = _cfg(tmp_path)
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    accept = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted")
    reject = _candidate("2501.00002", "2025-01-03T00:00:00Z", "Rejected")
    done = _candidate("2501.00003", "2025-01-04T00:00:00Z", "Done")
    _seed_accept_without_summary(db, accept, "2025-01")
    db.upsert_paper("2025-01", reject)
    db.upsert_triage(
        "2025-01",
        {
            "arxiv_id_base": reject["arxiv_id_base"],
            "decision": "reject",
            "confidence": 0.1,
            "reasons": ["nope"],
        },
    )
    db.upsert_paper("2025-01", done)
    db.upsert_triage(
        "2025-01",
        {
            "arxiv_id_base": done["arxiv_id_base"],
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["ok"],
        },
    )
    db.upsert_summary("2025-01", _summary_payload(done))
    assert db.get_accepted_without_summary() == [("2025-01", "2501.00001")]
    db.close()


def test_resummarize_stragglers_picks_up_accepted_without_summary(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    paper = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    _seed_accept_without_summary(db, paper, "2025-01")
    db.close()

    _patch_summary_stack(monkeypatch)
    stats = resummarize_stragglers(cfg, no_site=True)
    assert stats.attempted == 1
    assert stats.succeeded == 1
    assert stats.failed == 0
    assert stats.affected_months == ("2025-01",)

    db = DigestDB(cfg.data_dir / "digest.sqlite")
    assert db.get_summary("2501.00001") is not None
    assert db.get_accepted_without_summary() == []
    db.close()


def test_resummarize_stragglers_skips_papers_with_summary(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    paper = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    _seed_accept_without_summary(db, paper, "2025-01")
    db.upsert_summary("2025-01", _summary_payload(paper))
    db.close()

    calls = {"n": 0}

    def counting_summarize(paper, *_a, **_k):  # noqa: ANN001
        calls["n"] += 1
        return _summary_payload(paper)

    _patch_summary_stack(monkeypatch)
    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper", counting_summarize)
    stats = resummarize_stragglers(cfg, no_site=True)
    assert stats.attempted == 0
    assert stats.succeeded == 0
    assert calls["n"] == 0


def test_resummarize_stragglers_skips_rejected(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    paper = _candidate("2501.00002", "2025-01-03T00:00:00Z", "Rejected")
    db.upsert_paper("2025-01", paper)
    db.upsert_triage(
        "2025-01",
        {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": "reject",
            "confidence": 0.1,
            "reasons": ["nope"],
        },
    )
    db.close()

    _patch_summary_stack(monkeypatch)
    stats = resummarize_stragglers(cfg, no_site=True)
    assert stats == stats.__class__()
    assert stats.attempted == 0


def test_resummarize_stragglers_summarizes_from_abstract_on_pdf_error(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    paper = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    _seed_accept_without_summary(db, paper, "2025-01")
    db.close()

    _patch_summary_stack(monkeypatch, download_raises=True)
    stats = resummarize_stragglers(cfg, no_site=True)
    assert stats.attempted == 1
    assert stats.succeeded == 1
    assert stats.failed == 0
    assert stats.failed_ids == ()

    db = DigestDB(cfg.data_dir / "digest.sqlite")
    summary = db.get_summary("2501.00001")
    assert summary is not None
    assert summary["used_fulltext"] is False
    assert "abstract_only_fallback" in summary["notes"]
    db.close()


def test_resummarize_stragglers_honors_skip_ids(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    paper = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    _seed_accept_without_summary(db, paper, "2025-01")
    db.close()

    calls = {"n": 0}

    def counting_summarize(paper, *_a, **_k):  # noqa: ANN001
        calls["n"] += 1
        return _summary_payload(paper)

    _patch_summary_stack(monkeypatch)
    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper", counting_summarize)
    stats = resummarize_stragglers(cfg, no_site=True, skip_ids={"2501.00001"})
    assert stats.attempted == 0
    assert calls["n"] == 0


def test_run_window_skips_same_run_summary_failures_in_stragglers(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    paper = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    _seed_accept_without_summary(db, paper, "2025-01")
    db.close()

    monkeypatch.setattr(
        "eegfm_digest.pipeline.fetch_window_candidates",
        lambda *_a, **_k: [paper],
    )
    monkeypatch.setattr(
        "eegfm_digest.pipeline.run_month",
        lambda *_a, **_k: MonthRunStats(
            month="2025-01",
            candidates=1,
            accepted=1,
            summarized=0,
            summary_failures=1,
            failed_summary_ids=("2501.00001",),
        ),
    )
    calls = {"n": 0}

    def counting_resummarize(*_a, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        assert kwargs.get("skip_ids") == {"2501.00001"}
        return StragglerStats()

    monkeypatch.setattr("eegfm_digest.pipeline.resummarize_stragglers", counting_resummarize)
    since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    until = datetime(2025, 1, 3, tzinfo=timezone.utc)
    run_window(cfg, since, until, no_site=True)
    assert calls["n"] == 1


def test_rerender_month_preserves_existing_pdf_state(tmp_path):
    cfg = _cfg(tmp_path)
    month = "2025-01"
    paper = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    db.upsert_paper(month, paper)
    db.upsert_triage(
        month,
        {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["r1", "r2"],
        },
    )
    db.upsert_summary(month, _summary_payload(paper))
    db.close()

    month_out = cfg.output_dir / month
    month_out.mkdir(parents=True)
    existing_pdf = {
        "downloaded": True,
        "pdf_path": "outputs/2025-01/pdfs/2501.00001.pdf",
        "text_path": "outputs/2025-01/text/2501.00001.txt",
        "extract_meta": {"pages": 12},
    }
    (month_out / "backend_rows.jsonl").write_text(
        json.dumps(
            {
                "arxiv_id_base": "2501.00001",
                "pdf": existing_pdf,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (cfg.docs_dir).mkdir(parents=True, exist_ok=True)

    db = DigestDB(cfg.data_dir / "digest.sqlite")
    _rerender_month_from_db(cfg, db, month)
    db.close()

    rows = [
        json.loads(line)
        for line in (month_out / "backend_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["pdf"] == existing_pdf


def test_run_window_invokes_stragglers_sweep(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    paper = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    _seed_accept_without_summary(db, paper, "2025-01")
    db.close()

    monkeypatch.setattr(
        "eegfm_digest.pipeline.fetch_window_candidates",
        lambda *_a, **_k: [],
    )
    _patch_summary_stack(monkeypatch)

    since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    until = datetime(2025, 1, 2, tzinfo=timezone.utc)
    stats = run_window(cfg, since, until, no_site=True)
    assert stats.straggler_attempted == 1
    assert stats.straggler_succeeded == 1
    assert stats.straggler_failures == 0
    assert "2025-01" in stats.affected_months

    db = DigestDB(cfg.data_dir / "digest.sqlite")
    assert db.get_summary("2501.00001") is not None
    db.close()
