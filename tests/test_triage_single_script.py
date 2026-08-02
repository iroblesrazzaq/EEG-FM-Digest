from __future__ import annotations

import importlib.util
from pathlib import Path

from eegfm_digest.config import Config
from eegfm_digest.db import DigestDB
from eegfm_digest.stage_context import TriageStageContext


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "triage_single.py"
    spec = importlib.util.spec_from_file_location("triage_single_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paper() -> dict:
    return {
        "arxiv_id": "2401.12345v1",
        "arxiv_id_base": "2401.12345",
        "version": 1,
        "title": "New EEG FM Paper",
        "summary": "We pretrain a transferable EEG representation model.",
        "authors": ["Author One"],
        "categories": ["cs.LG"],
        "published": "2024-01-15T00:00:00Z",
        "updated": "2024-01-15T00:00:00Z",
        "links": {
            "abs": "https://arxiv.org/abs/2401.12345",
            "pdf": "https://arxiv.org/pdf/2401.12345.pdf",
        },
    }


def test_triage_single_returns_1_for_missing_paper(monkeypatch, capsys):
    module = _load_script_module()
    monkeypatch.setattr(module, "fetch_paper_by_id", lambda _arxiv_id: None)

    code = module.main(["9999.99999"])

    captured = capsys.readouterr()
    assert code == 1
    assert "No arXiv paper found" in captured.err


def test_triage_single_accept_persists_without_summary(monkeypatch, tmp_path, capsys):
    module = _load_script_module()
    paper = _paper()
    triage = {
        "arxiv_id_base": "2401.12345",
        "decision": "accept",
        "confidence": 0.91,
        "reasons": ["EEG is primary.", "Transferable pretraining."],
    }
    data_dir = tmp_path / "data"

    class _FakeLLM:
        def close(self) -> None:
            return None

    monkeypatch.setattr(module, "fetch_paper_by_id", lambda _arxiv_id: paper)
    monkeypatch.setattr(
        module,
        "load_config",
        lambda: Config(
            llm_provider="google",
            llm_model_triage="triage-model",
            llm_model_summary="summary-model",
            data_dir=data_dir,
            output_dir=tmp_path / "outputs",
        ),
    )
    monkeypatch.setattr(module, "load_api_key", lambda _provider=None: "test-key")
    monkeypatch.setattr(module, "build_llm_call", lambda _cfg: _FakeLLM())
    monkeypatch.setattr(
        module,
        "load_triage_stage_context",
        lambda _cfg: TriageStageContext(
            schema={"type": "object"},
            triage_prompt="triage",
            summarize_prompt="summarize",
            repair_prompt="repair",
            descriptor={
                "stage": "triage",
                "provider": "google",
                "model": "triage-model",
                "cache_version": "v1",
            },
        ),
    )
    monkeypatch.setattr(module, "triage_paper", lambda **_kwargs: triage)

    code = module.main(["2401.12345"])
    captured = capsys.readouterr()
    assert code == 0
    assert "**Decision:** `accept`" in captured.out
    assert "straggler" in captured.out.lower()

    db = DigestDB(data_dir / "digest.sqlite")
    assert db.get_triage("2401.12345")["decision"] == "accept"
    assert db.get_summary("2401.12345") is None
    assert db.get_accepted_without_summary() == [("2024-01", "2401.12345")]
    db.close()


def test_triage_single_skips_llm_when_already_triaged(monkeypatch, tmp_path, capsys):
    module = _load_script_module()
    paper = _paper()
    data_dir = tmp_path / "data"
    db = DigestDB(data_dir / "digest.sqlite")
    db.upsert_paper("2024-01", paper)
    db.upsert_triage(
        "2024-01",
        {
            "arxiv_id_base": "2401.12345",
            "decision": "reject",
            "confidence": 0.2,
            "reasons": ["not relevant"],
        },
    )
    db.close()

    called = {"n": 0}

    def boom(**_kwargs):  # noqa: ANN003
        called["n"] += 1
        raise AssertionError("should not call triage")

    monkeypatch.setattr(module, "fetch_paper_by_id", lambda _arxiv_id: paper)
    monkeypatch.setattr(
        module,
        "load_config",
        lambda: Config(
            llm_provider="google",
            llm_model_triage="triage-model",
            llm_model_summary="summary-model",
            data_dir=data_dir,
            output_dir=tmp_path / "outputs",
        ),
    )
    monkeypatch.setattr(module, "triage_paper", boom)

    code = module.main(["2401.12345"])
    captured = capsys.readouterr()
    assert code == 0
    assert called["n"] == 0
    assert "**Decision:** `reject`" in captured.out
    assert "Already triaged" in captured.out
