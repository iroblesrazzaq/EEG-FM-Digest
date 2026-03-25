import json
import sqlite3
from pathlib import Path

import pytest

from eegfm_digest.eval_triage import (
    build_gold_snapshot,
    compute_confusion,
    group_decision,
    main,
    score_gold_snapshot,
)


def _write_db(path: Path, rows: list[dict[str, str]]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE papers (
          arxiv_id_base TEXT PRIMARY KEY,
          month TEXT NOT NULL,
          metadata_json TEXT NOT NULL
        );
        CREATE TABLE triage (
          arxiv_id_base TEXT PRIMARY KEY,
          month TEXT NOT NULL,
          triage_json TEXT NOT NULL
        );
        """
    )
    for row in rows:
        metadata = {"arxiv_id": row["arxiv_id"], "title": row["title"]}
        triage = {"decision": row["decision"], "confidence": 0.9, "reasons": ["r1", "r2"]}
        conn.execute(
            "INSERT INTO papers(arxiv_id_base, month, metadata_json) VALUES (?, ?, ?)",
            (row["arxiv_id_base"], row["month"], json.dumps(metadata)),
        )
        conn.execute(
            "INSERT INTO triage(arxiv_id_base, month, triage_json) VALUES (?, ?, ?)",
            (row["arxiv_id_base"], row["month"], json.dumps(triage)),
        )
    conn.commit()
    conn.close()


def _gold_row(arxiv_id_base: str, title: str, gold_grouped: str) -> dict[str, str]:
    source_decision = "accept" if gold_grouped == "accept" else "reject"
    return {
        "arxiv_id_base": arxiv_id_base,
        "arxiv_id": f"{arxiv_id_base}v1",
        "title": title,
        "month": "2026-01",
        "gold_grouped": gold_grouped,
        "source_decision": source_decision,
        "notes": "",
    }


def test_group_decision_maps_borderline_to_not_pass():
    assert group_decision("accept") == "accept"
    assert group_decision("reject") == "not_pass"
    assert group_decision("borderline") == "not_pass"


def test_compute_confusion_matrix_counts():
    pairs = [
        ("accept", "accept"),
        ("accept", "not_pass"),
        ("not_pass", "accept"),
        ("not_pass", "not_pass"),
        ("accept", "accept"),
        ("not_pass", "not_pass"),
    ]
    counts = compute_confusion(pairs)
    assert counts == {"tp": 2, "fp": 1, "fn": 1, "tn": 2, "total": 6}


def test_build_gold_snapshot_exact_match(tmp_path):
    db_path = tmp_path / "digest.sqlite"
    _write_db(
        db_path,
        [
            {
                "month": "2026-01",
                "arxiv_id_base": "2601.00001",
                "arxiv_id": "2601.00001v1",
                "title": "Exact Match Title",
                "decision": "accept",
            }
        ],
    )
    titles_path = tmp_path / "titles.txt"
    titles_path.write_text("Exact Match Title\n", encoding="utf-8")
    out_path = tmp_path / "gold.jsonl"

    rows = build_gold_snapshot(db_path=db_path, titles_path=titles_path, out_path=out_path)

    assert len(rows) == 1
    assert rows[0]["arxiv_id_base"] == "2601.00001"
    assert rows[0]["gold_grouped"] == "accept"
    assert rows[0]["source_decision"] == "accept"
    assert out_path.exists()


def test_build_gold_snapshot_fails_on_ambiguous_fallback(tmp_path):
    db_path = tmp_path / "digest.sqlite"
    _write_db(
        db_path,
        [
            {
                "month": "2026-02",
                "arxiv_id_base": "2602.10001",
                "arxiv_id": "2602.10001v1",
                "title": "A Multi-decoder Neural Tracking Method for Accurately Predicting Speech Intelligibility",
                "decision": "reject",
            },
            {
                "month": "2026-02",
                "arxiv_id_base": "2602.10002",
                "arxiv_id": "2602.10002v1",
                "title": "Another Multi-decoder Neural Tracking Method for Accurately Predicting Speech Intelligibility",
                "decision": "reject",
            },
        ],
    )
    titles_path = tmp_path / "titles.txt"
    titles_path.write_text(
        "Multi-decoder Neural Tracking Method for Accurately Predicting Speech Intelligibility\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "gold.jsonl"

    with pytest.raises(RuntimeError, match="Ambiguous fallback title match"):
        build_gold_snapshot(db_path=db_path, titles_path=titles_path, out_path=out_path)


def test_score_returns_nonzero_when_gold_id_missing_from_db(tmp_path):
    db_path = tmp_path / "digest.sqlite"
    _write_db(
        db_path,
        [
            {
                "month": "2026-02",
                "arxiv_id_base": "2602.20001",
                "arxiv_id": "2602.20001v1",
                "title": "Present Paper",
                "decision": "accept",
            }
        ],
    )
    gold_path = tmp_path / "gold.jsonl"
    lines = [
        json.dumps(_gold_row("2602.20001", "Present Paper", "accept")),
        json.dumps(_gold_row("2602.29999", "Missing Paper", "not_pass")),
    ]
    gold_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = score_gold_snapshot(db_path=db_path, gold_path=gold_path)
    assert report["missing_ids"] == ["2602.29999"]

    exit_code = main(["score", "--gold", str(gold_path), "--db", str(db_path)])
    assert exit_code == 2
