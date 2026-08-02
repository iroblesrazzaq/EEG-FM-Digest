from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_sync_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sync_to_awesome.py"
    spec = importlib.util.spec_from_file_location("sync_to_awesome", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_backend_rows_filters_accepts_and_deduplicates(tmp_path, monkeypatch):
    module = _load_sync_module()
    month = "2025-01"
    output_dir = tmp_path / "outputs" / month
    output_dir.mkdir(parents=True)
    backend_path = output_dir / "backend_rows.jsonl"

    rows = [
        {
            "arxiv_id_base": "2501.00001",
            "title": "Accepted Paper",
            "published": "2025-01-02T00:00:00Z",
            "triage": {"decision": "accept"},
            "paper_summary": {"one_liner": "  Useful summary.  ", "published_date": "2025-01-02"},
        },
        {
            "arxiv_id_base": "2501.00001",
            "title": "Accepted Paper Duplicate",
            "published": "2025-01-02T00:00:00Z",
            "triage": {"decision": "accept"},
            "paper_summary": {"one_liner": "Duplicate row.", "published_date": "2025-01-02"},
        },
        {
            "arxiv_id_base": "2501.00002",
            "title": "Rejected Paper",
            "published": "2025-01-03T00:00:00Z",
            "triage": {"decision": "reject"},
            "paper_summary": None,
        },
    ]
    backend_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    papers = module.load_backend_rows(month)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2501.00001"
    assert papers[0].one_liner == "Useful summary."
    assert papers[0].year == "2025"
    assert module.format_markdown_entry(papers[0]) == (
        "- [Accepted Paper](https://arxiv.org/abs/2501.00001) - Useful summary. (2025)"
    )


def test_extract_arxiv_ids_matches_base_and_versioned_urls():
    module = _load_sync_module()
    readme = """
    - [Paper A](https://arxiv.org/abs/2501.00001v2) - Summary. (2025)
    - [Paper B](https://arxiv.org/pdf/2501.00002.pdf) - Summary. (2025)
    Mentioned as arXiv:2501.00003 in text.
    """

    found = module.extract_arxiv_ids(readme)

    assert found == {"2501.00001", "2501.00002", "2501.00003"}


def test_append_entries_preserves_spacing(tmp_path):
    module = _load_sync_module()
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# Awesome EEG FM\n", encoding="utf-8")

    module.append_entries(
        readme_path,
        [
            "- [Paper A](https://arxiv.org/abs/2501.00001) - Summary A. (2025)",
            "- [Paper B](https://arxiv.org/abs/2501.00002) - Summary B. (2025)",
        ],
    )

    assert readme_path.read_text(encoding="utf-8") == (
        "# Awesome EEG FM\n\n"
        "- [Paper A](https://arxiv.org/abs/2501.00001) - Summary A. (2025)\n"
        "- [Paper B](https://arxiv.org/abs/2501.00002) - Summary B. (2025)\n"
    )
