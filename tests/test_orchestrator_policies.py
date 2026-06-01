"""Characterization tests for intentional pipeline vs batch policy divergences."""

from __future__ import annotations

from eegfm_digest.batch import _parse_batch_config
from eegfm_digest.selection import select_papers_for_summary


def _candidate(arxiv_id_base: str, published: str) -> dict:
    return {
        "arxiv_id_base": arxiv_id_base,
        "published": published,
    }


def test_pipeline_borderline_policy_caps_borderline_pdfs():
    candidates = [
        _candidate("2501.00001", "2025-01-01T00:00:00Z"),
        _candidate("2501.00002", "2025-01-02T00:00:00Z"),
        _candidate("2501.00003", "2025-01-03T00:00:00Z"),
    ]
    triage_map = {
        "2501.00001": {"decision": "accept"},
        "2501.00002": {"decision": "borderline"},
        "2501.00003": {"decision": "borderline"},
    }
    pipeline_selected = select_papers_for_summary(
        candidates,
        triage_map,
        include_borderline=True,
        max_borderline_pdfs=1,
        max_accepted=10,
        borderline_policy="pipeline",
    )
    batch_selected = select_papers_for_summary(
        candidates,
        triage_map,
        include_borderline=True,
        max_borderline_pdfs=1,
        max_accepted=10,
        borderline_policy="batch",
    )
    assert [p["arxiv_id_base"] for p in pipeline_selected] == [
        "2501.00001",
        "2501.00002",
    ]
    assert [p["arxiv_id_base"] for p in batch_selected] == [
        "2501.00001",
        "2501.00002",
        "2501.00003",
    ]


def test_batch_config_env_path_defaults_empty(tmp_path):
    cfg_path = tmp_path / "batch.json"
    cfg_path.write_text('{"months": ["2025-01"]}', encoding="utf-8")
    cfg = _parse_batch_config(cfg_path)
    assert cfg.env_path == ""


def test_batch_config_env_path_optional(tmp_path):
    cfg_path = tmp_path / "batch.json"
    cfg_path.write_text(
        '{"months": ["2025-01"], "env_path": "/tmp/custom.env"}',
        encoding="utf-8",
    )
    cfg = _parse_batch_config(cfg_path)
    assert cfg.env_path == "/tmp/custom.env"


def test_documented_batch_vs_pipeline_divergences():
    """Explicit policy differences preserved by the refactor (not unified in Phase 8)."""
    divergences = {
        "borderline_cap": "pipeline uses max_borderline_pdfs; batch includes all borderline",
        "triage_failure": "pipeline skips DB row; batch persists synthetic reject",
        "summary_failure": "pipeline skips paper; batch fail-fast raises",
        "no_pdf": "pipeline supports --no-pdf; batch always downloads PDFs",
        "feature_paper": "pipeline supports --feature-paper; batch does not",
    }
    assert len(divergences) == 5
