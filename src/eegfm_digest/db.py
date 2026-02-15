from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class DigestDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
              arxiv_id_base TEXT PRIMARY KEY,
              month TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS triage (
              arxiv_id_base TEXT PRIMARY KEY,
              month TEXT NOT NULL,
              triage_json TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS summaries (
              arxiv_id_base TEXT PRIMARY KEY,
              month TEXT NOT NULL,
              summary_json TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS runs (
              month TEXT PRIMARY KEY,
              stats_json TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def upsert_paper(self, month: str, paper: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO papers(arxiv_id_base, month, metadata_json)
            VALUES (?, ?, ?)
            ON CONFLICT(arxiv_id_base) DO UPDATE SET
              month=excluded.month,
              metadata_json=excluded.metadata_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (paper["arxiv_id_base"], month, json.dumps(paper, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_triage(self, arxiv_id_base: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT triage_json FROM triage WHERE arxiv_id_base=?", (arxiv_id_base,)
        ).fetchone()
        return json.loads(row["triage_json"]) if row else None

    def upsert_triage(self, month: str, triage: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO triage(arxiv_id_base, month, triage_json)
            VALUES (?, ?, ?)
            ON CONFLICT(arxiv_id_base) DO UPDATE SET
              month=excluded.month,
              triage_json=excluded.triage_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (triage["arxiv_id_base"], month, json.dumps(triage, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_summary(self, arxiv_id_base: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT summary_json FROM summaries WHERE arxiv_id_base=?", (arxiv_id_base,)
        ).fetchone()
        return json.loads(row["summary_json"]) if row else None

    def upsert_summary(self, month: str, summary: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO summaries(arxiv_id_base, month, summary_json)
            VALUES (?, ?, ?)
            ON CONFLICT(arxiv_id_base) DO UPDATE SET
              month=excluded.month,
              summary_json=excluded.summary_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (summary["arxiv_id_base"], month, json.dumps(summary, ensure_ascii=False)),
        )
        self.conn.commit()

    def upsert_run(self, month: str, stats: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO runs(month, stats_json)
            VALUES (?, ?)
            ON CONFLICT(month) DO UPDATE SET
              stats_json=excluded.stats_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (month, json.dumps(stats, ensure_ascii=False)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
