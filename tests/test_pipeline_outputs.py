import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from eegfm_digest.config import Config
from eegfm_digest.llm import LLMRateLimitError
from eegfm_digest.pipeline import run_month


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


def _read_jsonl(path: Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_pipeline_writes_backend_rows_and_skips_site(monkeypatch, tmp_path):
    candidates = [
        _candidate("2501.00002", "2025-01-03T00:00:00Z", "Rejected Paper"),
        _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper"),
    ]

    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_args, **_kwargs: "test-key")

    class DummyLMCall:
        def close(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_args, **_kwargs: DummyLMCall())

    def fake_triage_paper(paper, *_args, **_kwargs):  # noqa: ANN001
        decision = "accept" if paper["arxiv_id_base"] == "2501.00001" else "reject"
        return {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": decision,
            "confidence": 0.9 if decision == "accept" else 0.1,
            "reasons": ["r1", "r2"],
        }

    monkeypatch.setattr("eegfm_digest.pipeline.triage_paper", fake_triage_paper)

    def fake_download_pdf(_url, out_path, _rate):  # noqa: ANN001
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4")
        return out_path

    def fake_extract_text(_pdf_path, text_path):  # noqa: ANN001
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(
            "Abstract\nEEG abstract\n\nIntroduction\nIntro\n\nMethods\nMethod\n\nResults\nResult\n\nConclusion\nEnd",
            encoding="utf-8",
        )
        return {"tool": "pypdf", "pages": 1, "chars": 100, "error": None}

    monkeypatch.setattr("eegfm_digest.pipeline.download_pdf", fake_download_pdf)
    monkeypatch.setattr("eegfm_digest.pipeline.extract_text", fake_extract_text)

    def fake_summarize_paper(paper, *_args, **_kwargs):  # noqa: ANN001
        return {
            "arxiv_id_base": paper["arxiv_id_base"],
            "title": paper["title"],
            "published_date": paper["published"][:10],
            "categories": paper["categories"],
            "paper_type": "method",
            "one_liner": "Concise summary line.",
            "detailed_summary": (
                "This work proposes a concise EEG modeling approach with explicit transfer framing "
                "and reports benchmark gains using pretrained representations. "
                "Its novel contribution is a deterministic pipeline that isolates method effects."
            ),
            "unique_contribution": "Deterministic contribution sentence.",
            "key_points": ["point one", "point two", "point three"],
            "data_scale": {
                "datasets": ["Dataset-A"],
                "subjects": 10,
                "eeg_hours": 2.0,
                "channels": 64,
            },
            "method": {
                "architecture": "Transformer",
                "objective": "Masked prediction",
                "pretraining": "Self-supervised",
                "finetuning": "Linear probe",
            },
            "evaluation": {
                "tasks": ["classification"],
                "benchmarks": ["Benchmark-A"],
                "headline_results": ["Improved AUROC"],
            },
            "open_source": {"code_url": None, "weights_url": None, "license": None},
            "tags": {
                "paper_type": ["eeg-fm"],
                "backbone": ["transformer"],
                "objective": ["masked-reconstruction"],
                "tokenization": ["time-patch"],
                "topology": ["fixed-montage"],
            },
            "limitations": ["limited cohorts", "single dataset"],
            "used_fulltext": True,
            "notes": "ok",
        }

    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper", fake_summarize_paper)

    cfg = Config(
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

    run_month(cfg, "2025-01", no_site=True)

    month_out = cfg.output_dir / "2025-01"
    backend_rows = _read_jsonl(month_out / "backend_rows.jsonl")
    triage_rows = _read_jsonl(month_out / "triage.jsonl")
    paper_rows = _read_jsonl(month_out / "papers.jsonl")
    digest = json.loads((month_out / "digest.json").read_text(encoding="utf-8"))

    assert [row["arxiv_id_base"] for row in backend_rows] == ["2501.00001", "2501.00002"]
    assert set(triage_rows[0].keys()) == {"arxiv_id_base", "decision", "confidence", "reasons"}
    assert len(paper_rows) == 1
    assert digest["featured_paper"] is None

    accepted_row = backend_rows[0]
    rejected_row = backend_rows[1]

    assert accepted_row["paper_summary"] is not None
    assert accepted_row["pdf"]["downloaded"] is True
    assert accepted_row["pdf"]["extract_meta"]["tool"] in {"pymupdf", "pypdf", "pdfminer"}

    assert rejected_row["paper_summary"] is None
    assert rejected_row["pdf"] == {
        "downloaded": False,
        "pdf_path": None,
        "text_path": None,
        "extract_meta": None,
    }

    assert not (tmp_path / "docs").exists()
    assert (cfg.output_dir / "2025-01" / "digest.json").exists()
    assert (cfg.data_dir / "digest.sqlite").exists()


def test_pipeline_writes_explicit_featured_paper_to_digest(monkeypatch, tmp_path):
    candidates = [_candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")]

    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_args, **_kwargs: "test-key")

    class DummyLMCall:
        def close(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_args, **_kwargs: DummyLMCall())
    monkeypatch.setattr(
        "eegfm_digest.pipeline.triage_paper",
        lambda paper, *_args, **_kwargs: {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["r1", "r2"],
        },
    )

    def fake_download_pdf(_url, out_path, _rate):  # noqa: ANN001
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4")
        return out_path

    def fake_extract_text(_pdf_path, text_path):  # noqa: ANN001
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("Abstract\nEEG abstract", encoding="utf-8")
        return {"tool": "pypdf", "pages": 1, "chars": 20, "error": None}

    monkeypatch.setattr("eegfm_digest.pipeline.download_pdf", fake_download_pdf)
    monkeypatch.setattr("eegfm_digest.pipeline.extract_text", fake_extract_text)
    monkeypatch.setattr(
        "eegfm_digest.pipeline.summarize_paper",
        lambda paper, *_args, **_kwargs: {
            "arxiv_id_base": paper["arxiv_id_base"],
            "title": paper["title"],
            "published_date": paper["published"][:10],
            "categories": paper["categories"],
            "paper_type": "method",
            "one_liner": "Concise summary line.",
            "detailed_summary": "Detailed summary.",
            "unique_contribution": "Deterministic contribution sentence.",
            "key_points": ["point one", "point two", "point three"],
            "data_scale": {"datasets": [], "subjects": None, "eeg_hours": None, "channels": None},
            "method": {"architecture": None, "objective": None, "pretraining": None, "finetuning": None},
            "evaluation": {"tasks": [], "benchmarks": [], "headline_results": []},
            "open_source": {"code_url": None, "weights_url": None, "license": None},
            "tags": {"paper_type": [], "backbone": [], "objective": [], "tokenization": [], "topology": []},
            "limitations": [],
            "used_fulltext": True,
            "notes": "ok",
        },
    )

    cfg = Config(
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

    run_month(cfg, "2025-01", no_site=True, feature_paper="2501.00001")

    digest = json.loads((cfg.output_dir / "2025-01" / "digest.json").read_text(encoding="utf-8"))
    assert digest["featured_paper"] == "2501.00001"


def test_pipeline_site_outputs_manifest_month_revision(monkeypatch, tmp_path):
    candidates = [
        _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper"),
    ]

    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_args, **_kwargs: "test-key")

    class DummyLMCall:
        def close(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_args, **_kwargs: DummyLMCall())

    monkeypatch.setattr(
        "eegfm_digest.pipeline.triage_paper",
        lambda paper, *_args, **_kwargs: {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["r1", "r2"],
        },
    )

    def fake_download_pdf(_url, out_path, _rate):  # noqa: ANN001
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4")
        return out_path

    def fake_extract_text(_pdf_path, text_path):  # noqa: ANN001
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(
            "Abstract\nEEG abstract\n\nIntroduction\nIntro\n\nMethods\nMethod\n\nResults\nResult\n\nConclusion\nEnd",
            encoding="utf-8",
        )
        return {"tool": "pypdf", "pages": 1, "chars": 100, "error": None}

    monkeypatch.setattr("eegfm_digest.pipeline.download_pdf", fake_download_pdf)
    monkeypatch.setattr("eegfm_digest.pipeline.extract_text", fake_extract_text)

    monkeypatch.setattr(
        "eegfm_digest.pipeline.summarize_paper",
        lambda paper, *_args, **_kwargs: {
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
            "data_scale": {
                "datasets": ["Dataset-A"],
                "subjects": 10,
                "eeg_hours": 2.0,
                "channels": 64,
            },
            "method": {
                "architecture": "Transformer",
                "objective": "Masked prediction",
                "pretraining": "Self-supervised",
                "finetuning": "Linear probe",
            },
            "evaluation": {
                "tasks": ["classification"],
                "benchmarks": ["Benchmark-A"],
                "headline_results": ["Improved AUROC"],
            },
            "open_source": {"code_url": None, "weights_url": None, "license": None},
            "tags": {
                "paper_type": ["eeg-fm"],
                "backbone": ["transformer"],
                "objective": ["masked-reconstruction"],
                "tokenization": ["time-patch"],
                "topology": ["fixed-montage"],
            },
            "limitations": ["limited cohorts", "single dataset"],
            "used_fulltext": True,
            "notes": "ok",
        },
    )

    cfg = Config(
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

    run_month(cfg, "2025-01", no_site=False)

    manifest_path = cfg.docs_dir / "data" / "months.json"
    month_payload_path = cfg.docs_dir / "digest" / "2025-01" / "papers.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    month_row = manifest["months"][0]
    expected_rev = hashlib.sha256(month_payload_path.read_bytes()).hexdigest()[:16]

    assert manifest["latest"] == "2025-01"
    assert month_row["month"] == "2025-01"
    assert month_row["month_rev"] == expected_rev
    assert (cfg.docs_dir / "index.html").exists()
    assert (cfg.docs_dir / "explore" / "index.html").exists()
    assert (cfg.docs_dir / "process" / "index.html").exists()


def test_pipeline_removes_stale_summary_when_triage_flips_to_reject(monkeypatch, tmp_path):
    candidate = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Flip Paper")
    candidates = [candidate]

    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_args, **_kwargs: "test-key")

    class DummyLMCall:
        def close(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_args, **_kwargs: DummyLMCall())

    triage_state = {"decision": "accept"}

    def fake_triage_paper(paper, *_args, **_kwargs):  # noqa: ANN001
        decision = triage_state["decision"]
        return {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": decision,
            "confidence": 0.9 if decision == "accept" else 0.1,
            "reasons": ["r1", "r2"],
        }

    monkeypatch.setattr("eegfm_digest.pipeline.triage_paper", fake_triage_paper)

    def fake_download_pdf(_url, out_path, _rate):  # noqa: ANN001
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4")
        return out_path

    def fake_extract_text(_pdf_path, text_path):  # noqa: ANN001
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("Abstract\nA\n\nIntroduction\nB\n\nMethods\nC\n\nResults\nD\n\nConclusion\nE", encoding="utf-8")
        return {"tool": "pypdf", "pages": 1, "chars": 100, "error": None}

    monkeypatch.setattr("eegfm_digest.pipeline.download_pdf", fake_download_pdf)
    monkeypatch.setattr("eegfm_digest.pipeline.extract_text", fake_extract_text)

    def fake_summarize_paper(paper, *_args, **_kwargs):  # noqa: ANN001
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
            "data_scale": {
                "datasets": ["Dataset-A"],
                "subjects": 10,
                "eeg_hours": 2.0,
                "channels": 64,
            },
            "method": {
                "architecture": "Transformer",
                "objective": "Masked prediction",
                "pretraining": "Self-supervised",
                "finetuning": "Linear probe",
            },
            "evaluation": {
                "tasks": ["classification"],
                "benchmarks": ["Benchmark-A"],
                "headline_results": ["Improved AUROC"],
            },
            "open_source": {"code_url": None, "weights_url": None, "license": None},
            "tags": {
                "paper_type": ["eeg-fm"],
                "backbone": ["transformer"],
                "objective": ["masked-reconstruction"],
                "tokenization": ["time-patch"],
                "topology": ["fixed-montage"],
            },
            "limitations": ["limited cohorts", "single dataset"],
            "used_fulltext": True,
            "notes": "ok",
        }

    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper", fake_summarize_paper)

    cfg = Config(
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

    run_month(cfg, "2025-01", no_site=True, force=True)

    triage_state["decision"] = "reject"
    run_month(cfg, "2025-01", no_site=True, force=True)

    month_out = cfg.output_dir / "2025-01"
    paper_rows = _read_jsonl(month_out / "papers.jsonl")
    backend_rows = _read_jsonl(month_out / "backend_rows.jsonl")

    assert paper_rows == []
    assert backend_rows[0]["paper_summary"] is None

    conn = sqlite3.connect(cfg.data_dir / "digest.sqlite")
    try:
        summary_count = conn.execute("SELECT COUNT(*) FROM summaries WHERE arxiv_id_base=?", ("2501.00001",)).fetchone()[0]
    finally:
        conn.close()
    assert summary_count == 0


def test_pipeline_reraises_llm_rate_limit_errors(monkeypatch, tmp_path):
    candidates = [_candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")]

    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_args, **_kwargs: "test-key")

    class DummyLMCall:
        def close(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_args, **_kwargs: DummyLMCall())
    monkeypatch.setattr(
        "eegfm_digest.pipeline.triage_paper",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMRateLimitError("429 upstream")),
    )

    cfg = Config(
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

    with pytest.raises(LLMRateLimitError):
        run_month(cfg, "2025-01", no_site=True)


def test_pipeline_pdf_download_failure_bumps_summary_failures(monkeypatch, tmp_path):
    candidates = [_candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")]

    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_args, **_kwargs: "test-key")

    class DummyLMCall:
        def close(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_args, **_kwargs: DummyLMCall())
    monkeypatch.setattr(
        "eegfm_digest.pipeline.triage_paper",
        lambda paper, *_a, **_k: {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["r"],
        },
    )

    def fail_download(_url, _out_path, _rate):  # noqa: ANN001
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr("eegfm_digest.pipeline.download_pdf", fail_download)

    summarize_called = False

    def fake_summarize(*_a, **_k):  # noqa: ANN001
        nonlocal summarize_called
        summarize_called = True
        raise AssertionError("summarize_paper should not be called when PDF fails")

    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper", fake_summarize)

    cfg = Config(
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

    stats = run_month(cfg, "2025-01", no_site=True)

    assert stats.summary_failures == 1
    assert stats.summarized == 0
    assert stats.accepted == 1
    assert summarize_called is False


def test_pipeline_missing_pdf_link_bumps_summary_failures(monkeypatch, tmp_path):
    paper = _candidate("2501.00001", "2025-01-02T00:00:00Z", "Accepted Paper")
    paper["links"]["pdf"] = ""  # accepted paper, no pdf URL
    candidates = [paper]

    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_args, **_kwargs: "test-key")

    class DummyLMCall:
        def close(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_args, **_kwargs: DummyLMCall())
    monkeypatch.setattr(
        "eegfm_digest.pipeline.triage_paper",
        lambda p, *_a, **_k: {
            "arxiv_id_base": p["arxiv_id_base"],
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["r"],
        },
    )
    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper", lambda *_a, **_k: pytest.fail("should not be called"))

    cfg = Config(
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

    stats = run_month(cfg, "2025-01", no_site=True)

    assert stats.summary_failures == 1
    assert stats.summarized == 0
    assert stats.accepted == 1
