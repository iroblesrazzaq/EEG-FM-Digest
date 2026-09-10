from __future__ import annotations

import json
from pathlib import Path

import pytest

from eegfm_digest.arch_export import (
    architecture_site_payload,
    build_export_payload,
    write_export_artifacts,
)
from eegfm_digest.arch_graph import merge_reve_config
from eegfm_digest.site import update_home
from eegfm_digest.site_templates import render_models_page, render_month_page

REVE_TITLE = (
    "REVE: A Foundation Model for EEG -- Adapting to Any Setup with Large-Scale "
    "Pretraining on 25,000 Subjects"
)


def _reve_payload() -> dict:
    return build_export_payload(
        arxiv_id="2510.21585",
        month="2025-10",
        title=REVE_TITLE,
        repo_id="brain-bzh/reve-base",
        cfg=merge_reve_config(None),
        tensor_names=None,
        source="hub_metadata+published_defaults",
        weights_url="https://huggingface.co/reve-model/reve",
    )


def test_build_export_payload_reve_has_graph_path_not_hfviewer():
    payload = _reve_payload()
    assert payload["graph_path"] == "data/arch/2510.21585.json"
    assert payload["label"] == "REVE"
    assert "hfviewer_url" not in payload
    site = architecture_site_payload(payload)
    assert site["graph_path"] == "data/arch/2510.21585.json"
    assert "hfviewer_url" not in site
    assert site["fact_sheet"]["num_layers"] == 22
    assert site["fact_sheet"]["hidden_size"] == 512
    assert payload["diagram"]["repeat"]["count"] == 22
    assert payload["diagram"]["title"] == "REVE"


def test_write_export_artifacts_patches_papers_and_catalog(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    month_dir = docs_dir / "digest" / "2025-10"
    month_dir.mkdir(parents=True)
    papers = {
        "featured_paper_id": "2510.21585",
        "month": "2025-10",
        "papers": [
            {
                "arxiv_id_base": "2510.21585",
                "title": REVE_TITLE,
                "summary": {"title": REVE_TITLE, "paper_type": "new_model"},
            }
        ],
        "stats": {"accepted": 1, "candidates": 1, "summarized": 1},
        "top_picks": [],
    }
    (month_dir / "papers.json").write_text(json.dumps(papers), encoding="utf-8")

    payload = _reve_payload()
    graph_path = write_export_artifacts(docs_dir, payload, patch_papers=True, refresh_shells=False)
    assert graph_path.exists()
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["nodes"][0]["id"] == "eeg"
    catalog = json.loads((docs_dir / "data" / "architectures.json").read_text(encoding="utf-8"))
    assert catalog["models"][0]["arxiv_id_base"] == "2510.21585"
    patched = json.loads((month_dir / "papers.json").read_text(encoding="utf-8"))
    arch = patched["papers"][0]["architecture"]
    assert arch["graph_path"] == "data/arch/2510.21585.json"
    assert "hfviewer_url" not in arch
    models_html = (docs_dir / "models" / "index.html").read_text(encoding="utf-8")
    assert "data-view='models'" in models_html
    assert "data/architectures.json" in models_html


def test_write_export_artifacts_does_not_write_when_paper_missing(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    payload = _reve_payload()
    with pytest.raises(FileNotFoundError):
        write_export_artifacts(docs_dir, payload, patch_papers=True, refresh_shells=False)
    assert not (docs_dir / "data" / "arch" / "2510.21585.json").exists()
    assert not (docs_dir / "data" / "architectures.json").exists()


def test_models_nav_and_month_shell_include_arch_graph_script():
    month_html = render_month_page("2025-10", [], {}, {})
    assert ">Models</a>" in month_html
    assert "arch-graph.js" in month_html
    assert "models/index.html" in month_html
    models_html = render_models_page()
    assert "data-view='models'" in models_html
    assert "arch-graph.js" in models_html
    assert 'class=\'site-nav-link active\'' in models_html or 'class="site-nav-link active"' in models_html


def test_update_home_writes_models_page(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    month_dir = docs_dir / "digest" / "2025-01"
    month_dir.mkdir(parents=True)
    (month_dir / "papers.json").write_text(
        json.dumps(
            {
                "month": "2025-01",
                "featured_paper_id": None,
                "stats": {"candidates": 0, "accepted": 0, "summarized": 0},
                "papers": [],
                "top_picks": [],
            }
        ),
        encoding="utf-8",
    )
    update_home(docs_dir)
    html = (docs_dir / "models" / "index.html").read_text(encoding="utf-8")
    assert "Models" in html
    home = (docs_dir / "index.html").read_text(encoding="utf-8")
    assert "models/index.html" in home
