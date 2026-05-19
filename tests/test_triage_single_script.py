from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eegfm_digest.config import Config


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "triage_single.py"
    spec = importlib.util.spec_from_file_location("triage_single_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_triage_single_returns_1_for_missing_paper(monkeypatch, capsys):
    module = _load_script_module()
    monkeypatch.setattr(module, "fetch_paper_by_id", lambda _arxiv_id: None)

    code = module.main(["9999.99999"])

    captured = capsys.readouterr()
    assert code == 1
    assert "No arXiv paper found" in captured.err
    assert captured.out == ""


def test_triage_single_uses_existing_review(monkeypatch, tmp_path, capsys):
    module = _load_script_module()
    output_dir = tmp_path / "outputs"
    month_dir = output_dir / "2025-01"
    month_dir.mkdir(parents=True)
    row = {
        "arxiv_id": "2401.12345v1",
        "arxiv_id_base": "2401.12345",
        "title": "Existing EEG FM Paper",
        "authors": ["Author One", "Author Two"],
        "links": {"abs": "https://arxiv.org/abs/2401.12345"},
        "triage": {
            "decision": "accept",
            "confidence": 0.91,
            "reasons": ["EEG is the primary modality.", "The abstract describes transferable pretraining."],
        },
    }
    (month_dir / "backend_rows.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "fetch_paper_by_id", lambda _arxiv_id: row)
    monkeypatch.setattr(
        module,
        "load_config",
        lambda: Config(llm_model_triage="triage-model", llm_model_summary="summary-model", output_dir=output_dir),
    )

    code = module.main(["2401.12345"])

    captured = capsys.readouterr()
    assert code == 0
    assert "Existing EEG FM Paper" in captured.out
    assert "**Decision:** `accept`" in captured.out
    assert "Reviewed In" in captured.out


def test_triage_single_runs_new_triage(monkeypatch, tmp_path, capsys):
    module = _load_script_module()
    output_dir = tmp_path / "outputs"
    paper = {
        "arxiv_id": "2401.12345v1",
        "arxiv_id_base": "2401.12345",
        "title": "New EEG FM Paper",
        "summary": "We pretrain a transferable EEG representation model.",
        "authors": ["Author One"],
        "links": {"abs": "https://arxiv.org/abs/2401.12345"},
    }
    triage = {
        "arxiv_id_base": "2401.12345",
        "decision": "borderline",
        "confidence": 0.62,
        "reasons": ["EEG is central.", "Transfer learning is explicit but FM scope is somewhat unclear."],
    }

    class _FakeLLM:
        def close(self) -> None:
            return None

    monkeypatch.setattr(module, "fetch_paper_by_id", lambda _arxiv_id: paper)
    monkeypatch.setattr(
        module,
        "load_config",
        lambda: Config(llm_model_triage="triage-model", llm_model_summary="summary-model", output_dir=output_dir),
    )
    monkeypatch.setattr(module, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(module, "build_llm_call", lambda _cfg: _FakeLLM())
    monkeypatch.setattr(module, "load_schema", lambda _path: {"type": "object"})
    monkeypatch.setattr(module, "triage_paper", lambda **_kwargs: triage)

    code = module.main(["2401.12345"])

    captured = capsys.readouterr()
    assert code == 0
    assert "New EEG FM Paper" in captured.out
    assert "**Decision:** `borderline`" in captured.out
    assert "[2401.12345](https://arxiv.org/abs/2401.12345)" in captured.out
