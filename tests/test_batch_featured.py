"""Batch runner featured-paper config (configs/featured_papers.json)."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eegfm_digest.batch import (
    BatchRunConfig,
    _run_summary_phase_for_month,
    _run_summary_phase_for_month_with_featured_guard,
    load_featured_papers_map,
)
from eegfm_digest.config import Config
from eegfm_digest.llm import LLMCallConfig
from eegfm_digest.render import build_digest


def test_load_featured_papers_map_reads_json(tmp_path: Path) -> None:
    p = tmp_path / "featured.json"
    p.write_text(
        json.dumps({"2025-01": "2501.00001", "2025-02": None, "2025-03": "  "}),
        encoding="utf-8",
    )
    m = load_featured_papers_map(p)
    assert m["2025-01"] == "2501.00001"
    assert m["2025-02"] is None
    assert m["2025-03"] is None


def test_load_featured_papers_map_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_featured_papers_map(tmp_path / "nope.json") == {}


def test_load_featured_papers_map_rejects_non_object(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid featured papers"):
        load_featured_papers_map(p)


def _minimal_summary(arxiv_id_base: str = "2501.00001") -> dict:
    return {
        "arxiv_id_base": arxiv_id_base,
        "title": "Accepted Paper Title Here Long Enough",
        "published_date": "2025-01-02",
        "categories": ["cs.LG"],
        "paper_type": "method",
        "one_liner": "A concise summary line meeting minimum length reqs.",
        "detailed_summary": (
            "This work proposes an EEG modeling approach with transfer framing "
            "and reports benchmark gains. The contribution is explicit and reproducible."
        ),
        "unique_contribution": "Deterministic pipeline isolating method effects.",
        "key_points": ["point one", "point two", "point three"],
        "data_scale": {
            "datasets": ["Dataset-A"],
            "subjects": 10,
            "eeg_hours": 2.0,
            "channels": 64,
        },
        "method": {
            "architecture": "Transformer",
            "foundation_model_related": False,
            "pretraining_signals": [],
            "finetuning": "supervised",
        },
        "evaluation": {"tasks": ["classification"], "metrics": ["accuracy"], "baselines": []},
        "open_source": {"code": "unknown", "weights": "unknown", "datasets": False},
        "tags": [],
        "limitations": [],
        "used_fulltext": True,
        "notes": "test",
    }


def test_summary_phase_passes_featured_to_build_digest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    month = "2025-01"
    aid = "2501.00001"
    month_out = tmp_path / "outputs" / month
    month_out.mkdir(parents=True, exist_ok=True)
    candidates = [
        {
            "arxiv_id": f"{aid}v1",
            "arxiv_id_base": aid,
            "version": 1,
            "title": "Accepted Paper",
            "summary": "Abstract text here.",
            "authors": ["A"],
            "categories": ["cs.LG"],
            "published": "2025-01-02T00:00:00Z",
            "updated": "2025-01-02T00:00:00Z",
            "links": {"abs": f"https://arxiv.org/abs/{aid}", "pdf": f"https://arxiv.org/pdf/{aid}.pdf"},
        },
    ]
    triage_rows = [
        {"arxiv_id_base": aid, "decision": "accept", "confidence": 0.9, "reasons": ["r1"]},
    ]
    (month_out / "arxiv_raw.json").write_text(json.dumps(candidates), encoding="utf-8")
    lines = "\n".join(json.dumps(r) for r in triage_rows)
    (month_out / "triage.jsonl").write_text(lines + "\n", encoding="utf-8")

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
    run_cfg = BatchRunConfig(
        months=[month],
        months_from_outputs=False,
        no_site=True,
        sync_cache_from_outputs=False,
    )

    db = MagicMock()
    summary = _minimal_summary(aid)
    db.get_summary_with_meta.return_value = {"data": summary, "meta": {"cache_version": "x"}}

    featured_kw: list[str | None] = []

    def capture_build_digest(*args, **kwargs):
        featured_kw.append(kwargs.get("featured_paper"))
        return build_digest(*args, **kwargs)

    monkeypatch.setattr(
        "eegfm_digest.batch.build_digest",
        capture_build_digest,
    )
    monkeypatch.setattr("eegfm_digest.batch.is_cache_current", lambda _meta, _cv: True)

    class DummyLm:
        def close(self) -> None:  # noqa: ANN201
            pass

    llm = DummyLm()
    llm_config = LLMCallConfig(
        provider="google",
        api_key="k",
        model="m",
        temperature=0.2,
        max_output_tokens=100,
        base_url=None,
    )

    _run_summary_phase_for_month(
        cfg,
        run_cfg,
        month,
        db,
        llm,
        llm_config,
        featured_paper=aid,
    )

    assert featured_kw == [aid]
    digest = json.loads((month_out / "digest.json").read_text(encoding="utf-8"))
    assert digest.get("featured_paper") == aid


def test_summary_phase_featured_guard_retries_without_stale_featured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    run_cfg = BatchRunConfig(months=["2025-01"], no_site=True)
    llm_config = LLMCallConfig(
        provider="google",
        api_key="k",
        model="m",
        temperature=0.2,
        max_output_tokens=100,
        base_url=None,
    )
    calls: list[str | None] = []

    def fake_summary_phase(*_args, featured_paper=None, **_kwargs) -> None:
        calls.append(featured_paper)
        if featured_paper == "2501.99999":
            raise RuntimeError(
                "Featured paper 2501.99999 was not accepted for this month; "
                "only accepted papers can be featured."
            )

    monkeypatch.setattr("eegfm_digest.batch._run_summary_phase_for_month", fake_summary_phase)

    _run_summary_phase_for_month_with_featured_guard(
        cfg,
        run_cfg,
        "2025-01",
        MagicMock(),
        MagicMock(),
        llm_config,
        featured_paper="2501.99999",
    )

    assert calls == ["2501.99999", None]
    assert "continuing without featured paper" in capsys.readouterr().out


def test_summary_phase_featured_guard_reraises_unrelated_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    run_cfg = BatchRunConfig(months=["2025-01"], no_site=True)
    llm_config = LLMCallConfig(
        provider="google",
        api_key="k",
        model="m",
        temperature=0.2,
        max_output_tokens=100,
        base_url=None,
    )

    def fake_summary_phase(*_args, **_kwargs) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr("eegfm_digest.batch._run_summary_phase_for_month", fake_summary_phase)

    with pytest.raises(RuntimeError, match="database is locked"):
        _run_summary_phase_for_month_with_featured_guard(
            cfg,
            run_cfg,
            "2025-01",
            MagicMock(),
            MagicMock(),
            llm_config,
            featured_paper="2501.99999",
        )
