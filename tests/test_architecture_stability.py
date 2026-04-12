from __future__ import annotations

import json
import sqlite3
import sys

from eegfm_digest.batch import run_batch
from eegfm_digest.config import Config, DEFAULT_OPENROUTER_MODEL
from eegfm_digest.pipeline import run_month
from eegfm_digest.run import main as run_main
from eegfm_digest.site import update_home, write_month_site


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
        "links": {
            "abs": f"https://arxiv.org/abs/{arxiv_id_base}",
            "pdf": f"https://arxiv.org/pdf/{arxiv_id_base}.pdf",
        },
    }


def _summary(paper: dict) -> dict:
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
        "data_scale": {"datasets": ["Dataset-A"], "subjects": 10, "eeg_hours": 2.0, "channels": 64},
        "method": {
            "architecture": "Transformer",
            "objective": "Masked prediction",
            "pretraining": "Self-supervised",
            "finetuning": "Linear probe",
        },
        "evaluation": {"tasks": ["classification"], "benchmarks": ["Benchmark-A"], "headline_results": ["AUROC"]},
        "open_source": {"code_url": None, "weights_url": None, "license": None},
        "tags": {
            "paper_type": ["new-model"],
            "backbone": ["transformer"],
            "objective": ["masked-reconstruction"],
            "tokenization": ["time-patch"],
            "topology": ["fixed-montage"],
        },
        "limitations": ["limited cohorts", "single dataset"],
        "used_fulltext": True,
        "notes": "ok",
    }


def _backend_rows(papers: list[dict], summary_map: dict[str, dict | None]) -> list[dict]:
    rows: list[dict] = []
    for paper in papers:
        rows.append(
            {
                "arxiv_id": paper["arxiv_id"],
                "arxiv_id_base": paper["arxiv_id_base"],
                "title": paper["title"],
                "published": paper["published"],
                "authors": paper["authors"],
                "categories": paper["categories"],
                "links": paper["links"],
                "triage": {"decision": "accept", "confidence": 0.9, "reasons": ["fit"]},
                "paper_summary": summary_map.get(paper["arxiv_id_base"]),
                "pdf": {"downloaded": True, "pdf_path": "a", "text_path": "b", "extract_meta": {"tool": "cached"}},
            }
        )
    return rows


def test_single_month_run_main_hard_pins_stepflash(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "eegfm_digest.run.load_config",
        lambda: Config(
            llm_model_triage="env-triage",
            llm_model_summary="env-summary",
            output_dir=tmp_path / "outputs",
            data_dir=tmp_path / "data",
            docs_dir=tmp_path / "docs",
        ),
    )
    monkeypatch.setattr(
        "eegfm_digest.run.run_month",
        lambda cfg, month, **kwargs: captured.update({"cfg": cfg, "month": month, "kwargs": kwargs}),
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--month", "2026-03"])

    run_main()

    cfg = captured["cfg"]
    assert isinstance(cfg, Config)
    assert cfg.llm_model_triage == DEFAULT_OPENROUTER_MODEL
    assert cfg.llm_model_summary == DEFAULT_OPENROUTER_MODEL
    assert captured["month"] == "2026-03"
    assert captured["kwargs"]["topic"] == "eeg-fm"


def test_batch_run_respects_configured_models(monkeypatch, tmp_path):
    config_path = tmp_path / "batch.json"
    config_path.write_text(
        json.dumps(
            {
                "months": [],
                "months_from_outputs": False,
                "triage_model": "batch-triage",
                "summary_model": "batch-summary",
            }
        ),
        encoding="utf-8",
    )

    captured_models: list[str] = []

    class DummyLLM:
        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "eegfm_digest.batch.load_config",
        lambda: Config(
            llm_model_triage="default-triage",
            llm_model_summary="default-summary",
            output_dir=tmp_path / "outputs",
            data_dir=tmp_path / "data",
            docs_dir=tmp_path / "docs",
        ),
    )
    monkeypatch.setattr("eegfm_digest.batch.load_api_key", lambda: "test-key")
    monkeypatch.setattr("eegfm_digest.batch.load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("eegfm_digest.batch._effective_months", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "eegfm_digest.batch.build_llm_call",
        lambda cfg: captured_models.append(cfg.model) or DummyLLM(),
    )

    run_batch(config_path)

    assert captured_models == ["batch-triage", "batch-summary"]


def test_write_month_site_resolves_featured_override_and_null(tmp_path):
    docs_dir = tmp_path / "docs"
    month = "2026-02"
    papers = [
        _candidate("2602.10000", "2026-02-01T00:00:00Z", "Other Paper"),
        _candidate("2602.18478", "2026-02-09T00:00:00Z", "ZUNA: Flexible EEG Superresolution with Position-Aware Diffusion Autoencoders"),
    ]
    summaries = [_summary(papers[0]), _summary(papers[1])]
    digest = {
        "month": month,
        "stats": {"candidates": 2, "accepted": 2, "summarized": 2},
        "top_picks": ["2602.10000"],
        "sections": [],
    }

    write_month_site(
        docs_dir=docs_dir,
        month=month,
        summaries=summaries,
        metadata={paper["arxiv_id_base"]: paper for paper in papers},
        digest=digest,
        backend_rows=_backend_rows(papers, {paper["arxiv_id_base"]: summary for paper, summary in zip(papers, summaries)}),
        featured_overrides={"2026-02": "2602.18478", "2026-03": None},
    )
    update_home(docs_dir, featured_overrides={"2026-02": "2602.18478", "2026-03": None})

    payload = json.loads((docs_dir / "digest" / "2026-02" / "papers.json").read_text(encoding="utf-8"))
    manifest = json.loads((docs_dir / "data" / "months.json").read_text(encoding="utf-8"))

    assert payload["featured_paper_id"] == "2602.18478"
    assert manifest["months"][0]["featured"]["arxiv_id_base"] == "2602.18478"

    march = "2026-03"
    write_month_site(
        docs_dir=docs_dir,
        month=march,
        summaries=summaries[:1],
        metadata={papers[0]["arxiv_id_base"]: papers[0]},
        digest={
            "month": march,
            "stats": {"candidates": 1, "accepted": 1, "summarized": 1},
            "top_picks": ["2602.10000"],
            "sections": [],
        },
        backend_rows=_backend_rows([papers[0]], {papers[0]["arxiv_id_base"]: summaries[0]}),
        featured_overrides={"2026-02": "2602.18478", "2026-03": None},
    )
    update_home(docs_dir, featured_overrides={"2026-02": "2602.18478", "2026-03": None})

    march_payload = json.loads((docs_dir / "digest" / "2026-03" / "papers.json").read_text(encoding="utf-8"))
    manifest = json.loads((docs_dir / "data" / "months.json").read_text(encoding="utf-8"))
    month_map = {row["month"]: row for row in manifest["months"]}

    assert "featured_paper_id" in march_payload
    assert march_payload["featured_paper_id"] is None
    assert month_map["2026-03"]["featured"] is None


def test_update_home_rejects_invalid_featured_override(tmp_path):
    docs_dir = tmp_path / "docs"
    month = "2026-02"
    papers = [_candidate("2602.10000", "2026-02-01T00:00:00Z", "Other Paper")]
    summaries = [_summary(papers[0])]

    write_month_site(
        docs_dir=docs_dir,
        month=month,
        summaries=summaries,
        metadata={papers[0]["arxiv_id_base"]: papers[0]},
        digest={
            "month": month,
            "stats": {"candidates": 1, "accepted": 1, "summarized": 1},
            "top_picks": ["2602.10000"],
            "sections": [],
        },
        backend_rows=_backend_rows(papers, {papers[0]["arxiv_id_base"]: summaries[0]}),
        featured_overrides={},
    )

    try:
        update_home(docs_dir, featured_overrides={"2026-02": "2602.18478"})
    except RuntimeError as exc:
        assert "missing paper id 2602.18478" in str(exc)
    else:
        raise AssertionError("Expected invalid featured override to raise")


def test_pipeline_migrates_old_db_rows_and_treats_missing_metadata_as_stale(monkeypatch, tmp_path):
    candidate = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    db_path = tmp_path / "data" / "digest.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE papers (
              arxiv_id_base TEXT PRIMARY KEY,
              month TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE triage (
              arxiv_id_base TEXT PRIMARY KEY,
              month TEXT NOT NULL,
              triage_json TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE summaries (
              arxiv_id_base TEXT PRIMARY KEY,
              month TEXT NOT NULL,
              summary_json TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE runs (
              month TEXT PRIMARY KEY,
              stats_json TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            "INSERT INTO triage(arxiv_id_base, month, triage_json) VALUES (?, ?, ?)",
            ("2501.00001", "2025-01", json.dumps({"arxiv_id_base": "2501.00001", "decision": "accept", "confidence": 0.1, "reasons": ["old"]})),
        )
        conn.execute(
            "INSERT INTO summaries(arxiv_id_base, month, summary_json) VALUES (?, ?, ?)",
            ("2501.00001", "2025-01", json.dumps(_summary(candidate))),
        )
        conn.commit()
    finally:
        conn.close()

    triage_calls = {"count": 0}
    summary_calls = {"count": 0}

    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_args, **_kwargs: [candidate])
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda: "test-key")

    class DummyLLM:
        def close(self) -> None:
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_args, **_kwargs: DummyLLM())
    monkeypatch.setattr("eegfm_digest.pipeline.download_pdf", lambda _url, out_path, _rate: out_path.parent.mkdir(parents=True, exist_ok=True) or out_path.write_bytes(b"%PDF"))

    def fake_extract_text(_pdf_path, text_path):
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("Abstract\nEEG\n\nMethods\nM", encoding="utf-8")
        return {"tool": "pypdf", "pages": 1, "chars": 20, "error": None}

    monkeypatch.setattr("eegfm_digest.pipeline.extract_text", fake_extract_text)

    def fake_triage(*_args, **_kwargs):
        triage_calls["count"] += 1
        return (
            {"arxiv_id_base": "2501.00001", "decision": "accept", "confidence": 0.9, "reasons": ["fresh"]},
            {"repair_used": False},
        )

    def fake_summary(paper, *_args, **_kwargs):
        summary_calls["count"] += 1
        return (_summary(paper), {"repair_used": True})

    monkeypatch.setattr("eegfm_digest.pipeline.triage_paper_with_meta", fake_triage)
    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper_with_meta", fake_summary)

    cfg = Config(
        llm_model_triage="triage-model",
        llm_model_summary="summary-model",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
        arxiv_rate_limit_seconds=0.0,
        pdf_rate_limit_seconds=0.0,
    )

    run_month(cfg, "2025-01", no_site=True)

    assert triage_calls["count"] == 1
    assert summary_calls["count"] == 1

    conn = sqlite3.connect(db_path)
    try:
        triage_cols = {row[1] for row in conn.execute("PRAGMA table_info(triage)").fetchall()}
        summary_cols = {row[1] for row in conn.execute("PRAGMA table_info(summaries)").fetchall()}
        triage_meta = json.loads(
            conn.execute("SELECT triage_meta_json FROM triage WHERE arxiv_id_base='2501.00001'").fetchone()[0]
        )
        summary_meta = json.loads(
            conn.execute("SELECT summary_meta_json FROM summaries WHERE arxiv_id_base='2501.00001'").fetchone()[0]
        )
    finally:
        conn.close()

    assert "triage_meta_json" in triage_cols
    assert "summary_meta_json" in summary_cols
    assert triage_meta["cache_version"]
    assert summary_meta["repair_used"] is True


def test_pipeline_cache_versions_reuse_and_invalidate(monkeypatch, tmp_path):
    candidate = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    triage_calls = {"count": 0}
    summary_calls = {"count": 0}
    prompts = {
        "prompts/triage.md": "triage-a {{TITLE}} {{ABSTRACT}}",
        "prompts/summarize.md": "summary-a {{INPUT_JSON}}",
        "prompts/repair_json.md": "repair {{SCHEMA_JSON}} {{BAD_OUTPUT}}",
    }

    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_args, **_kwargs: [candidate])
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda: "test-key")
    monkeypatch.setattr("eegfm_digest.pipeline._read", lambda path: prompts[path])

    class DummyLLM:
        def close(self) -> None:
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_args, **_kwargs: DummyLLM())
    monkeypatch.setattr("eegfm_digest.pipeline.download_pdf", lambda _url, out_path, _rate: out_path.parent.mkdir(parents=True, exist_ok=True) or out_path.write_bytes(b"%PDF"))

    def fake_extract_text(_pdf_path, text_path):
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("Abstract\nEEG\n\nMethods\nM", encoding="utf-8")
        return {"tool": "pypdf", "pages": 1, "chars": 20, "error": None}

    monkeypatch.setattr("eegfm_digest.pipeline.extract_text", fake_extract_text)

    def fake_triage(*_args, **_kwargs):
        triage_calls["count"] += 1
        return (
            {"arxiv_id_base": "2501.00001", "decision": "accept", "confidence": 0.9, "reasons": ["fresh"]},
            {"repair_used": False},
        )

    def fake_summary(paper, *_args, **_kwargs):
        summary_calls["count"] += 1
        return (_summary(paper), {"repair_used": True})

    monkeypatch.setattr("eegfm_digest.pipeline.triage_paper_with_meta", fake_triage)
    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper_with_meta", fake_summary)

    cfg = Config(
        llm_model_triage="triage-model",
        llm_model_summary="summary-model",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
        arxiv_rate_limit_seconds=0.0,
        pdf_rate_limit_seconds=0.0,
    )

    run_month(cfg, "2025-01", no_site=True)
    run_month(cfg, "2025-01", no_site=True)
    assert triage_calls["count"] == 1
    assert summary_calls["count"] == 1

    prompts["prompts/triage.md"] = "triage-b {{TITLE}} {{ABSTRACT}}"
    run_month(cfg, "2025-01", no_site=True)
    assert triage_calls["count"] == 2
    assert summary_calls["count"] == 1

    prompts["prompts/summarize.md"] = "summary-b {{INPUT_JSON}}"
    run_month(cfg, "2025-01", no_site=True)
    assert triage_calls["count"] == 2
    assert summary_calls["count"] == 2

    conn = sqlite3.connect(cfg.data_dir / "digest.sqlite")
    try:
        summary_meta = json.loads(
            conn.execute("SELECT summary_meta_json FROM summaries WHERE arxiv_id_base='2501.00001'").fetchone()[0]
        )
    finally:
        conn.close()

    assert summary_meta["repair_used"] is True
