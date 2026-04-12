from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from eegfm_digest.config import Config
from eegfm_digest.csv_export import CSV_COLUMNS, export_all_csv
from eegfm_digest.run import main as run_main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _backend_row(
    arxiv_id_base: str,
    *,
    decision: str = "accept",
    published: str = "2025-01-02T00:00:00Z",
    with_summary: bool = True,
) -> dict:
    summary = None
    if with_summary:
        summary = {
            "one_liner": "EEG FM summary line.",
            "tags": {
                "paper_type": ["new-model"],
                "backbone": ["transformer"],
                "objective": ["masked-reconstruction"],
                "tokenization": ["time-patch"],
                "topology": ["fixed-montage"],
            },
        }

    return {
        "arxiv_id": f"{arxiv_id_base}v2",
        "arxiv_id_base": arxiv_id_base,
        "version": 2,
        "title": f"Paper {arxiv_id_base}",
        "summary": "abstract",
        "authors": ["Author A", "Author B"],
        "categories": ["cs.LG"],
        "published": published,
        "updated": published,
        "links": {
            "abs": f"https://arxiv.org/abs/{arxiv_id_base}",
            "pdf": f"https://arxiv.org/pdf/{arxiv_id_base}.pdf",
        },
        "triage": {
            "decision": decision,
            "confidence": 0.93,
            "reasons": ["fit"],
        },
        "paper_summary": summary,
        "pdf": {
            "downloaded": with_summary,
            "pdf_path": None,
            "text_path": None,
            "extract_meta": None,
        },
    }


def test_run_csv_flag_exports_expected_schema(monkeypatch, tmp_path):
    month = "2025-01"
    output_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        "eegfm_digest.run.load_config",
        lambda: Config(
            llm_model_triage="triage-model",
            llm_model_summary="summary-model",
            output_dir=output_dir,
            data_dir=tmp_path / "data",
            docs_dir=tmp_path / "docs",
        ),
    )

    def fake_run_month(_cfg, run_month_value, **_kwargs):  # noqa: ANN001
        assert run_month_value == month
        _write_jsonl(
            output_dir / month / "backend_rows.jsonl",
            [
                _backend_row("2501.00001"),
                _backend_row("2501.00002", decision="reject"),
                _backend_row("2501.00003", with_summary=False),
            ],
        )

    monkeypatch.setattr("eegfm_digest.run.run_month", fake_run_month)
    monkeypatch.setattr(sys, "argv", ["run.py", "--month", month, "--csv"])

    run_main()

    fieldnames, rows = _read_csv_rows(output_dir / month / "digest.csv")

    assert fieldnames == CSV_COLUMNS
    assert rows == [
        {
            "arxiv_id": "2501.00001v2",
            "title": "Paper 2501.00001",
            "published": "2025-01-02T00:00:00Z",
            "authors": "Author A;Author B",
            "decision": "accept",
            "confidence": "0.93",
            "one_liner": "EEG FM summary line.",
            "tags": "new-model,transformer,masked-reconstruction,time-patch,fixed-montage",
            "arxiv_url": "https://arxiv.org/abs/2501.00001",
        },
        {
            "arxiv_id": "2501.00003v2",
            "title": "Paper 2501.00003",
            "published": "2025-01-02T00:00:00Z",
            "authors": "Author A;Author B",
            "decision": "accept",
            "confidence": "0.93",
            "one_liner": "",
            "tags": "",
            "arxiv_url": "https://arxiv.org/abs/2501.00003",
        },
    ]


def test_export_all_csv_aggregates_all_months(tmp_path):
    outputs_dir = tmp_path / "outputs"
    _write_jsonl(outputs_dir / "2025-01" / "backend_rows.jsonl", [_backend_row("2501.00001")])
    _write_jsonl(
        outputs_dir / "2025-02" / "backend_rows.jsonl",
        [_backend_row("2502.00001", published="2025-02-03T00:00:00Z")],
    )

    export_all_csv(outputs_dir)

    fieldnames, rows = _read_csv_rows(outputs_dir / "all_papers.csv")

    assert fieldnames == CSV_COLUMNS
    assert [row["arxiv_id"] for row in rows] == ["2501.00001v2", "2502.00001v2"]
