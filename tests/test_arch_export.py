from __future__ import annotations

import json
from pathlib import Path

import pytest

from eegfm_digest.arch_export import (
    architecture_site_payload,
    build_export_payload,
    config_filenames,
    export_all_digest,
    iter_digest_hf_papers,
    safetensors_filenames,
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
    assert ">Model Gallery</a>" in month_html
    assert "arch-graph.js" in month_html
    assert "models/index.html" in month_html
    models_html = render_models_page()
    assert "<h1>Model Gallery</h1>" in models_html
    assert "<title>EEG-FM Digest | Model Gallery</title>" in models_html
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
    assert "Model Gallery" in html
    home = (docs_dir / "index.html").read_text(encoding="utf-8")
    assert "models/index.html" in home


def test_export_all_digest_exports_owner_name_skips_org_page(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    month_dir = docs_dir / "digest" / "2025-10"
    month_dir.mkdir(parents=True)
    papers = {
        "featured_paper_id": "2501.00001",
        "month": "2025-10",
        "papers": [
            {
                "arxiv_id_base": "2501.00001",
                "title": "Tiny LM",
                "summary": {
                    "title": "Tiny LM",
                    "open_source": {"weights_url": "https://huggingface.co/acme/tiny-lm"},
                },
            },
            {
                "arxiv_id_base": "2601.06134",
                "title": "DeeperBrain",
                "summary": {
                    "title": "DeeperBrain",
                    "open_source": {"weights_url": "https://huggingface.co/DeeperBrain"},
                },
            },
        ],
        "stats": {"accepted": 2, "candidates": 2, "summarized": 2},
        "top_picks": [],
    }
    (month_dir / "papers.json").write_text(json.dumps(papers), encoding="utf-8")

    found = iter_digest_hf_papers(docs_dir)
    assert [row["repo"] for row in found] == ["acme/tiny-lm"]

    calls: list[dict] = []

    def fake_export(**kwargs):
        calls.append(kwargs)
        payload = {
            "arxiv_id_base": kwargs["arxiv_id"],
            "title": kwargs.get("title") or kwargs["arxiv_id"],
            "label": "Tiny LM",
            "month": kwargs["month"],
            "hf_repo": kwargs["repo"],
            "source": "test",
            "graph_path": f"data/arch/{kwargs['arxiv_id']}.json",
            "fact_sheet": {},
            "nodes": [{"id": "input", "label": "Input", "kind": "input"}],
            "edges": [],
            "diagram": {"title": "Tiny LM"},
        }
        write_export_artifacts(
            kwargs["docs_dir"],
            payload,
            patch_papers=True,
            refresh_shells=False,
        )
        return payload

    result = export_all_digest(docs_dir, exporter=fake_export)
    assert [row["arxiv_id"] for row in result["exported"]] == ["2501.00001"]
    assert result["skipped"] == []
    assert len(calls) == 1
    assert calls[0]["fallback_repo"] is None
    assert (docs_dir / "data" / "arch" / "2501.00001.json").exists()
    assert not (docs_dir / "data" / "arch" / "2601.06134.json").exists()
    patched = json.loads((month_dir / "papers.json").read_text(encoding="utf-8"))
    by_id = {row["arxiv_id_base"]: row for row in patched["papers"]}
    assert by_id["2501.00001"]["architecture"]["graph_path"] == "data/arch/2501.00001.json"
    assert "graph_path" not in (by_id["2601.06134"].get("architecture") or {})


def test_export_all_digest_records_gated_skips(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    month_dir = docs_dir / "digest" / "2026-01"
    month_dir.mkdir(parents=True)
    papers = {
        "month": "2026-01",
        "papers": [
            {
                "arxiv_id_base": "2601.00001",
                "title": "Open",
                "summary": {"open_source": {"weights_url": "https://huggingface.co/acme/open"}},
            },
            {
                "arxiv_id_base": "2601.00002",
                "title": "Gated",
                "summary": {"open_source": {"weights_url": "https://huggingface.co/acme/gated"}},
            },
        ],
    }
    (month_dir / "papers.json").write_text(json.dumps(papers), encoding="utf-8")

    def fake_export(**kwargs):
        if kwargs["repo"] == "acme/gated":
            raise PermissionError("gated")
        payload = {
            "arxiv_id_base": kwargs["arxiv_id"],
            "title": "Open",
            "label": "Open",
            "month": kwargs["month"],
            "hf_repo": kwargs["repo"],
            "source": "test",
            "graph_path": f"data/arch/{kwargs['arxiv_id']}.json",
            "fact_sheet": {},
            "nodes": [],
            "edges": [],
        }
        write_export_artifacts(kwargs["docs_dir"], payload, patch_papers=True, refresh_shells=False)
        return payload

    result = export_all_digest(docs_dir, exporter=fake_export)
    assert [row["arxiv_id"] for row in result["exported"]] == ["2601.00001"]
    assert result["skipped"][0]["arxiv_id"] == "2601.00002"
    assert "PermissionError" in result["skipped"][0]["reason"]
    assert (docs_dir / "data" / "arch" / "2601.00001.json").exists()
    assert not (docs_dir / "data" / "arch" / "2601.00002.json").exists()


def test_daily_workflow_does_not_call_arch_export():
    text = Path(".github/workflows/daily-digest.yml").read_text(encoding="utf-8")
    assert "arch_export" not in text
    assert "all-digest" not in text


def test_all_digest_rejects_single_export_flags():
    from eegfm_digest.arch_export import main

    with pytest.raises(SystemExit):
        main(["--all-digest", "--arxiv", "2510.21585", "--repo", "a/b", "--month", "2025-10"])


def test_iter_digest_hf_papers_uses_known_hub_map(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    month_dir = docs_dir / "digest" / "2025-10"
    month_dir.mkdir(parents=True)
    papers = {
        "month": "2025-10",
        "papers": [
            {
                "arxiv_id_base": "2510.22257",
                "title": "LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis",
                "summary": {"open_source": {"weights_url": None}},
            }
        ],
    }
    (month_dir / "papers.json").write_text(json.dumps(papers), encoding="utf-8")
    found = iter_digest_hf_papers(docs_dir)
    assert found == [
        {
            "month": "2025-10",
            "arxiv_id": "2510.22257",
            "repo": "PulpBio/LUNA",
            "title": "LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis",
        }
    ]


def test_config_and_safetensors_prefer_base_over_huge():
    info = {
        "siblings": [
            {"rfilename": "TUAB/FEMBA_tiny.safetensors"},
            {"rfilename": "TUAB/FEMBA_base.safetensors"},
            {"rfilename": "LUNA_huge.safetensors"},
            {"rfilename": "LUNA_base.safetensors"},
            {"rfilename": "classifier/config.json"},
            {"rfilename": "base/model_cfg.json"},
            {"rfilename": "config.json"},
            {"rfilename": "braintokenizer/model_cfg.json"},
        ]
    }
    assert config_filenames(info)[0] == "config.json"
    assert "base/model_cfg.json" in config_filenames(info)
    assert config_filenames(info).index("base/model_cfg.json") < config_filenames(info).index(
        "braintokenizer/model_cfg.json"
    )
    names = safetensors_filenames(info)
    assert names[0] == "LUNA_base.safetensors"
    assert names[1] == "LUNA_huge.safetensors"
    assert names.index("TUAB/FEMBA_base.safetensors") < names.index("TUAB/FEMBA_tiny.safetensors")
