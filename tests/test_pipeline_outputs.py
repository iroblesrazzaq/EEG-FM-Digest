import json
from pathlib import Path

from eegfm_digest.config import Config
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
    monkeypatch.setattr("eegfm_digest.pipeline.load_api_key", lambda: "test-key")

    class DummyGemini:
        def __init__(self, _config):  # noqa: ANN001
            return None

    monkeypatch.setattr("eegfm_digest.pipeline.GeminiClient", DummyGemini)

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
        gemini_model_triage="triage-model",
        gemini_model_summary="summary-model",
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

    assert [row["arxiv_id_base"] for row in backend_rows] == ["2501.00001", "2501.00002"]
    assert set(triage_rows[0].keys()) == {"arxiv_id_base", "decision", "confidence", "reasons"}
    assert len(paper_rows) == 1

    accepted_row = backend_rows[0]
    rejected_row = backend_rows[1]

    assert accepted_row["paper_summary"] is not None
    assert accepted_row["pdf"]["downloaded"] is True
    assert accepted_row["pdf"]["extract_meta"]["tool"] == "pypdf"

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
