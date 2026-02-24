import json
from pathlib import Path

from eegfm_digest.site import render_month_page, update_home, write_month_site


def test_month_page_snapshot():
    digest = {"top_picks": ["2501.00001"], "sections": [{"title": "new_model", "paper_ids": ["2501.00001"]}]}
    summaries = [
        {
            "arxiv_id_base": "2501.00001",
            "title": "EEG FM One",
            "published_date": "2025-01-04",
            "paper_type": "new_model",
            "one_liner": "A concise line",
            "detailed_summary": (
                "This paper proposes a reusable EEG foundation model backbone for cross-task transfer. "
                "Its key novelty is a compact architecture that preserves performance while reducing compute."
            ),
            "key_points": [
                "Introduces an EEG foundation model for cross-task transfer.",
                "Uses a compact backbone to reduce runtime cost.",
                "Shows competitive performance across benchmarks.",
            ],
            "unique_contribution": "A unique thing",
            "tags": {
                "paper_type": ["eeg-fm"],
                "backbone": ["transformer"],
                "objective": ["masked-reconstruction"],
                "tokenization": ["time-patch"],
                "topology": ["fixed-montage"],
            },
            "open_source": {"code_url": "https://code", "weights_url": None, "license": None},
        }
    ]
    metadata = {
        "2501.00001": {
            "authors": ["Alice", "Bob"],
            "links": {"abs": "https://arxiv.org/abs/2501.00001"},
        }
    }
    html = render_month_page("2025-01", summaries, metadata, digest)
    snapshot = Path("tests/fixtures/month_page_snapshot.html").read_text(encoding="utf-8")
    assert html.strip() == snapshot.strip()


def test_write_month_site_payload_includes_summary_failures(tmp_path):
    docs_dir = tmp_path / "docs"
    month = "2025-01"
    summaries = [
        {
            "arxiv_id_base": "2501.00001",
            "title": "Accepted With Summary",
            "published_date": "2025-01-01",
            "categories": ["cs.LG"],
            "key_points": ["p1", "p2"],
            "unique_contribution": "u",
            "detailed_summary": "d" * 90,
            "one_liner": "l",
            "paper_type": "new_model",
            "data_scale": {"datasets": [], "subjects": None, "eeg_hours": None, "channels": None},
            "method": {"architecture": None, "objective": None, "pretraining": None, "finetuning": None},
            "evaluation": {"tasks": [], "benchmarks": [], "headline_results": []},
            "open_source": {"code_url": None, "weights_url": None, "license": None},
            "tags": {"paper_type": [], "backbone": [], "objective": [], "tokenization": [], "topology": []},
            "limitations": ["l1", "l2"],
            "used_fulltext": True,
            "notes": "ok",
        }
    ]
    metadata = {
        "2501.00001": {
            "authors": ["Alice"],
            "links": {"abs": "https://arxiv.org/abs/2501.00001"},
        }
    }
    digest = {
        "month": month,
        "stats": {"candidates": 3, "accepted": 2, "summarized": 1},
        "top_picks": ["2501.00001"],
        "sections": [],
    }
    backend_rows = [
        {
            "arxiv_id": "2501.00001v1",
            "arxiv_id_base": "2501.00001",
            "title": "Accepted With Summary",
            "published": "2025-01-01T00:00:00Z",
            "authors": ["Alice"],
            "categories": ["cs.LG"],
            "links": {"abs": "https://arxiv.org/abs/2501.00001"},
            "triage": {"decision": "accept", "confidence": 0.9, "reasons": ["fit"]},
            "paper_summary": summaries[0],
            "pdf": {"downloaded": True, "pdf_path": "a", "text_path": "b", "extract_meta": {"tool": "cached"}},
        },
        {
            "arxiv_id": "2501.00002v1",
            "arxiv_id_base": "2501.00002",
            "title": "Accepted But Failed Summary",
            "published": "2025-01-02T00:00:00Z",
            "authors": ["Bob"],
            "categories": ["cs.LG"],
            "links": {"abs": "https://arxiv.org/abs/2501.00002"},
            "triage": {"decision": "accept", "confidence": 0.7, "reasons": ["fit"]},
            "paper_summary": None,
            "pdf": {
                "downloaded": False,
                "pdf_path": None,
                "text_path": None,
                "extract_meta": {"error": "download_or_extract_failed:ClientError"},
            },
        },
        {
            "arxiv_id": "2501.00003v1",
            "arxiv_id_base": "2501.00003",
            "title": "Rejected",
            "published": "2025-01-03T00:00:00Z",
            "authors": ["Carol"],
            "categories": ["cs.LG"],
            "links": {"abs": "https://arxiv.org/abs/2501.00003"},
            "triage": {"decision": "reject", "confidence": 0.1, "reasons": ["no"]},
            "paper_summary": None,
            "pdf": {"downloaded": False, "pdf_path": None, "text_path": None, "extract_meta": None},
        },
    ]

    write_month_site(
        docs_dir=docs_dir,
        month=month,
        summaries=summaries,
        metadata=metadata,
        digest=digest,
        backend_rows=backend_rows,
    )

    payload = json.loads((docs_dir / "digest" / month / "papers.json").read_text(encoding="utf-8"))
    assert payload["month"] == "2025-01"
    assert payload["stats"] == {"candidates": 3, "accepted": 2, "summarized": 1}
    assert len(payload["papers"]) == 2
    failed = [row for row in payload["papers"] if row["arxiv_id_base"] == "2501.00002"][0]
    assert failed["summary"] is None
    assert failed["summary_failed_reason"] == "download_or_extract_failed:ClientError"


def test_update_home_writes_month_manifest(tmp_path):
    docs_dir = tmp_path / "docs"
    month_a = docs_dir / "digest" / "2025-01"
    month_b = docs_dir / "digest" / "2025-02"
    month_a.mkdir(parents=True, exist_ok=True)
    month_b.mkdir(parents=True, exist_ok=True)
    (month_a / "papers.json").write_text(
        json.dumps(
            {
                "month": "2025-01",
                "stats": {"candidates": 3, "accepted": 0, "summarized": 0},
                "papers": [],
                "top_picks": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (month_b / "papers.json").write_text(
        json.dumps(
            {
                "month": "2025-02",
                "stats": {"candidates": 2, "accepted": 1, "summarized": 1},
                "papers": [{"arxiv_id_base": "2502.00001", "summary": {"title": "x"}}],
                "top_picks": ["2502.00001"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    update_home(docs_dir)

    manifest = json.loads((docs_dir / "data" / "months.json").read_text(encoding="utf-8"))
    assert manifest["latest"] == "2025-02"
    assert [row["month"] for row in manifest["months"]] == ["2025-02", "2025-01"]
    month_map = {row["month"]: row for row in manifest["months"]}
    assert month_map["2025-01"]["empty_state"] == "no_accepts"
    assert month_map["2025-02"]["empty_state"] == "has_papers"
