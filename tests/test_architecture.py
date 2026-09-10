from __future__ import annotations

import json
from pathlib import Path

import httpx

from eegfm_digest.architecture import (
    architecture_from_summary,
    fact_sheet_from_config,
    parse_hf_repo_id,
)
from eegfm_digest.config import Config
from eegfm_digest.pipeline import run_month


def test_parse_hf_repo_id_model_urls():
    assert parse_hf_repo_id("https://huggingface.co/reve-model/reve") == "reve-model/reve"
    assert parse_hf_repo_id("https://hf.co/org/model") == "org/model"
    assert parse_hf_repo_id("https://huggingface.co/org/model/tree/main") == "org/model"
    assert parse_hf_repo_id("org/model") == "org/model"


def test_parse_hf_repo_id_rejects_non_model_refs():
    assert parse_hf_repo_id("https://huggingface.co/DeeperBrain") is None
    assert parse_hf_repo_id("https://github.com/org/repo") is None
    assert parse_hf_repo_id("HuggingFace Hub") is None
    assert parse_hf_repo_id(None) is None
    assert parse_hf_repo_id("") is None
    assert parse_hf_repo_id("https://huggingface.co/models/bert") is None


def test_fact_sheet_from_config_reads_common_keys():
    sheet = fact_sheet_from_config(
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 2048,
            "vocab_size": 32000,
        }
    )
    assert sheet["model_type"] == "llama"
    assert sheet["architectures"] == "LlamaForCausalLM"
    assert sheet["hidden_size"] == 512
    assert sheet["num_layers"] == 12
    assert sheet["num_attention_heads"] == 8
    assert sheet["num_key_value_heads"] == 2
    assert sheet["context_length"] == 2048
    assert sheet["vocab_size"] == 32000


def _new_model_summary(*, weights_url: str | None, paper_type: str = "new_model") -> dict:
    return {
        "arxiv_id_base": "2501.00001",
        "paper_type": paper_type,
        "open_source": {"code_url": None, "weights_url": weights_url, "license": None},
        "used_fulltext": True,
        "notes": "ok",
    }


def test_architecture_skips_non_new_model():
    assert architecture_from_summary(_new_model_summary(weights_url="https://huggingface.co/org/m", paper_type="method")) is None


def test_architecture_skips_org_page_without_fetch():
    result = architecture_from_summary(
        _new_model_summary(weights_url="https://huggingface.co/DeeperBrain"),
        fetch_config=lambda _repo: (_ for _ in ()).throw(AssertionError("should not fetch")),
    )
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "not_a_model_repo"


def test_architecture_skips_github_weights():
    result = architecture_from_summary(
        _new_model_summary(weights_url="https://github.com/org/weights"),
        fetch_config=lambda _repo: (_ for _ in ()).throw(AssertionError("should not fetch")),
    )
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "not_huggingface"


def test_architecture_ok_from_config():
    result = architecture_from_summary(
        _new_model_summary(weights_url="https://huggingface.co/org/model"),
        fetch_config=lambda _repo: {"model_type": "gpt2", "n_embd": 768, "n_layer": 12, "n_head": 12},
    )
    assert result["status"] == "ok"
    assert result["hf_repo"] == "org/model"
    assert result["hfviewer_url"] == "https://hfviewer.com/org/model"
    assert result["fact_sheet"]["hidden_size"] == 768
    assert result["fact_sheet"]["num_layers"] == 12


def test_architecture_skips_gated_and_missing(monkeypatch):
    def gated(_repo: str):
        raise PermissionError("gated:403")

    gated_result = architecture_from_summary(
        _new_model_summary(weights_url="https://huggingface.co/org/gated"),
        fetch_config=gated,
    )
    assert gated_result["status"] == "skipped"
    assert gated_result["skip_reason"] == "gated"
    assert gated_result["hfviewer_url"] == "https://hfviewer.com/org/gated"

    missing_result = architecture_from_summary(
        _new_model_summary(weights_url="https://huggingface.co/org/missing"),
        fetch_config=lambda _repo: (_ for _ in ()).throw(FileNotFoundError("missing_config")),
    )
    assert missing_result["skip_reason"] == "missing_config"


def test_architecture_reuses_ok_existing():
    existing = {
        "status": "ok",
        "hf_repo": "org/model",
        "hfviewer_url": "https://hfviewer.com/org/model",
        "fact_sheet": {"hidden_size": 1},
        "skip_reason": None,
    }
    result = architecture_from_summary(
        _new_model_summary(weights_url="https://huggingface.co/org/model"),
        existing,
        fetch_config=lambda _repo: (_ for _ in ()).throw(AssertionError("should reuse")),
    )
    assert result is existing


def _candidate(arxiv_id_base: str) -> dict:
    return {
        "arxiv_id": f"{arxiv_id_base}v1",
        "arxiv_id_base": arxiv_id_base,
        "version": 1,
        "title": "New EEG FM",
        "summary": "abstract",
        "authors": ["Author A"],
        "categories": ["cs.LG"],
        "published": "2025-01-02T00:00:00Z",
        "updated": "2025-01-02T00:00:00Z",
        "links": {
            "abs": f"https://arxiv.org/abs/{arxiv_id_base}",
            "pdf": f"https://arxiv.org/pdf/{arxiv_id_base}.pdf",
        },
    }


def _summary(paper: dict, *, weights_url: str) -> dict:
    return {
        "arxiv_id_base": paper["arxiv_id_base"],
        "title": paper["title"],
        "published_date": paper["published"][:10],
        "categories": paper["categories"],
        "paper_type": "new_model",
        "one_liner": "Concise summary line.",
        "detailed_summary": (
            "This work proposes a concise EEG modeling approach with explicit transfer framing "
            "and reports benchmark gains using pretrained representations."
        ),
        "unique_contribution": "Deterministic contribution sentence.",
        "key_points": ["point one", "point two", "point three"],
        "data_scale": {"datasets": [], "subjects": None, "eeg_hours": None, "channels": None},
        "method": {"architecture": "Transformer", "objective": None, "pretraining": None, "finetuning": None},
        "evaluation": {"tasks": [], "benchmarks": [], "headline_results": []},
        "open_source": {"code_url": None, "weights_url": weights_url, "license": None},
        "tags": {
            "paper_type": ["new-model"],
            "backbone": [],
            "objective": [],
            "tokenization": [],
            "topology": [],
        },
        "limitations": [],
        "used_fulltext": True,
        "notes": "ok",
    }


def _stub_pipeline(monkeypatch, candidate: dict, summary: dict) -> None:
    monkeypatch.setattr("eegfm_digest.pipeline.fetch_month_candidates", lambda *_a, **_k: [candidate])
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda *_a, **_k: "test-key")

    class DummyLMCall:
        def close(self):
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.build_llm_call", lambda *_a, **_k: DummyLMCall())
    monkeypatch.setattr(
        "eegfm_digest.pipeline.triage_paper",
        lambda paper, *_a, **_k: {
            "arxiv_id_base": paper["arxiv_id_base"],
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["ok"],
        },
    )

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
    monkeypatch.setattr("eegfm_digest.pipeline.summarize_paper", lambda paper, *_a, **_k: summary)


def test_pipeline_architecture_ok_does_not_fail_run(monkeypatch, tmp_path: Path):
    candidate = _candidate("2501.00001")
    summary = _summary(candidate, weights_url="https://huggingface.co/org/model")
    _stub_pipeline(monkeypatch, candidate, summary)
    monkeypatch.setattr(
        "eegfm_digest.architecture.fetch_hf_config",
        lambda repo_id: {"model_type": "llama", "hidden_size": 256, "num_hidden_layers": 6},
    )
    cfg = Config(
        llm_model_triage="triage-model",
        llm_model_summary="summary-model",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
        arxiv_rate_limit_seconds=0.0,
        pdf_rate_limit_seconds=0.0,
    )
    stats = run_month(cfg, "2025-01", no_site=False)
    assert stats.summary_failures == 0
    payload = json.loads((tmp_path / "docs" / "digest" / "2025-01" / "papers.json").read_text(encoding="utf-8"))
    architecture = payload["papers"][0]["architecture"]
    assert architecture["status"] == "ok"
    assert architecture["fact_sheet"]["hidden_size"] == 256
    assert architecture["hfviewer_url"] == "https://hfviewer.com/org/model"


def test_pipeline_architecture_http_error_still_green(monkeypatch, tmp_path: Path):
    candidate = _candidate("2501.00001")
    summary = _summary(candidate, weights_url="https://huggingface.co/org/model")
    _stub_pipeline(monkeypatch, candidate, summary)

    def boom(_repo: str):
        raise httpx.HTTPStatusError(
            "403",
            request=httpx.Request("GET", "https://huggingface.co/org/model/resolve/main/config.json"),
            response=httpx.Response(403, request=httpx.Request("GET", "https://example.com")),
        )

    monkeypatch.setattr("eegfm_digest.architecture.fetch_hf_config", boom)
    cfg = Config(
        llm_model_triage="triage-model",
        llm_model_summary="summary-model",
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
        arxiv_rate_limit_seconds=0.0,
        pdf_rate_limit_seconds=0.0,
    )
    stats = run_month(cfg, "2025-01", no_site=True)
    assert stats.summary_failures == 0
    rows = [
        json.loads(line)
        for line in (cfg.output_dir / "2025-01" / "backend_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["architecture"]["status"] == "skipped"
    assert rows[0]["architecture"]["skip_reason"] == "fetch_failed"


def test_site_js_renders_architecture_controls():
    site_js = Path("docs/assets/site.js").read_text(encoding="utf-8")
    assert "function renderArchitectureFactSheet(architecture)" in site_js
    assert "View architecture" in site_js
    assert "arch-fact-sheet" in site_js
    style = Path("docs/assets/style.css").read_text(encoding="utf-8")
    assert ".resource-btn-arch" in style


def test_batch_cache_hit_keeps_sqlite_architecture(monkeypatch, tmp_path: Path):
    from dataclasses import replace

    from eegfm_digest.batch import BatchRunConfig, _run_summary_phase_for_month
    from eegfm_digest.db import DigestDB
    from eegfm_digest.llm import LLMCallConfig

    month = "2025-01"
    candidate = _candidate("2501.00001")
    summary = _summary(candidate, weights_url="https://huggingface.co/org/model")
    architecture = {
        "status": "ok",
        "hf_repo": "org/model",
        "hfviewer_url": "https://hfviewer.com/org/model",
        "fact_sheet": {"hidden_size": 256, "model_type": "llama"},
        "skip_reason": None,
    }
    month_out = tmp_path / "outputs" / month
    month_out.mkdir(parents=True)
    (month_out / "arxiv_raw.json").write_text(json.dumps([candidate]), encoding="utf-8")
    (month_out / "triage.jsonl").write_text(
        json.dumps(
            {
                "arxiv_id_base": "2501.00001",
                "decision": "accept",
                "confidence": 0.9,
                "reasons": ["ok"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = replace(
        Config(
            llm_provider="google",
            llm_model_triage="m",
            llm_model_summary="m",
        ),
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
    )
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    db.upsert_paper(month, candidate)
    db.upsert_triage(
        month,
        {
            "arxiv_id_base": "2501.00001",
            "decision": "accept",
            "confidence": 0.9,
            "reasons": ["ok"],
        },
    )
    db.upsert_summary(month, summary, meta={"cache_version": "x", "architecture": architecture})
    db.close()

    monkeypatch.setattr("eegfm_digest.batch.is_cache_current", lambda *_a, **_k: True)

    class DummyLm:
        def close(self) -> None:
            return None

    db = DigestDB(cfg.data_dir / "digest.sqlite")
    _run_summary_phase_for_month(
        cfg,
        BatchRunConfig(months=[month], months_from_outputs=False, no_site=True),
        month,
        db,
        DummyLm(),
        LLMCallConfig(
            provider="google",
            api_key="k",
            model="m",
            temperature=0.2,
            max_output_tokens=100,
            base_url=None,
        ),
    )
    db.close()

    rows = [
        json.loads(line)
        for line in (month_out / "backend_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["architecture"]["hfviewer_url"] == "https://hfviewer.com/org/model"
    assert rows[0]["architecture"]["fact_sheet"]["hidden_size"] == 256