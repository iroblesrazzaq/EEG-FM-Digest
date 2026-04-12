from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "arxiv_id",
    "title",
    "published",
    "authors",
    "decision",
    "confidence",
    "one_liner",
    "tags",
    "arxiv_url",
]

_TAG_FIELD_ORDER = ["paper_type", "backbone", "objective", "tokenization", "topology"]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _list_to_delimited(value: Any, delimiter: str) -> str:
    if not isinstance(value, list):
        return ""
    return delimiter.join(str(item) for item in value if str(item).strip())


def _flatten_tags(summary: dict[str, Any] | None) -> str:
    if not isinstance(summary, dict):
        return ""
    tags = summary.get("tags")
    if not isinstance(tags, dict):
        return ""

    flattened: list[str] = []
    for field in _TAG_FIELD_ORDER:
        values = tags.get(field, [])
        if not isinstance(values, list):
            continue
        for value in values:
            tag = str(value).strip()
            if tag and tag not in flattened:
                flattened.append(tag)
    return ",".join(flattened)


def csv_row_from_backend_row(row: dict[str, Any]) -> dict[str, str] | None:
    triage = row.get("triage")
    if not isinstance(triage, dict) or triage.get("decision") != "accept":
        return None

    summary = row.get("paper_summary")
    links = row.get("links")
    arxiv_id_base = str(row.get("arxiv_id_base", "")).strip()
    arxiv_url = ""
    if isinstance(links, dict):
        arxiv_url = str(links.get("abs", "")).strip()
    if not arxiv_url and arxiv_id_base:
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id_base}"

    return {
        "arxiv_id": str(row.get("arxiv_id", "")).strip(),
        "title": str(row.get("title", "")).strip(),
        "published": str(row.get("published", "")).strip(),
        "authors": _list_to_delimited(row.get("authors"), ";"),
        "decision": str(triage.get("decision", "")).strip(),
        "confidence": str(triage.get("confidence", "")),
        "one_liner": str(summary.get("one_liner", "")).strip() if isinstance(summary, dict) else "",
        "tags": _flatten_tags(summary),
        "arxiv_url": arxiv_url,
    }


def csv_rows_from_backend_rows(backend_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for backend_row in backend_rows:
        csv_row = csv_row_from_backend_row(backend_row)
        if csv_row is not None:
            rows.append(csv_row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def export_month_csv(month_out: Path, destination: Path | None = None) -> Path:
    backend_rows = _load_jsonl(month_out / "backend_rows.jsonl")
    rows = csv_rows_from_backend_rows(backend_rows)
    output_path = destination or (month_out / "digest.csv")
    _write_csv(output_path, rows)
    return output_path


def export_all_csv(outputs_dir: Path, destination: Path | None = None) -> Path:
    all_rows: list[dict[str, str]] = []
    for month_out in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        all_rows.extend(csv_rows_from_backend_rows(_load_jsonl(month_out / "backend_rows.jsonl")))

    all_rows.sort(key=lambda row: (row["published"], row["arxiv_id"]))
    output_path = destination or (outputs_dir / "all_papers.csv")
    _write_csv(output_path, all_rows)
    return output_path
