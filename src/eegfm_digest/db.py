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
              triage_meta_json TEXT,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS summaries (
              arxiv_id_base TEXT PRIMARY KEY,
              month TEXT NOT NULL,
              summary_json TEXT NOT NULL,
              summary_meta_json TEXT,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS runs (
              month TEXT PRIMARY KEY,
              stats_json TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._ensure_column("triage", "triage_meta_json", "TEXT")
        self._ensure_column("summaries", "summary_meta_json", "TEXT")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def upsert_paper(self, month: str, paper: dict[str, Any], cache_key: str | None = None) -> None:
        cache_key = cache_key or paper["arxiv_id_base"]
        self.conn.execute(
            """
            INSERT INTO papers(arxiv_id_base, month, metadata_json)
            VALUES (?, ?, ?)
            ON CONFLICT(arxiv_id_base) DO UPDATE SET
              month=excluded.month,
              metadata_json=excluded.metadata_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (cache_key, month, json.dumps(paper, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_triage(self, arxiv_id_base: str, cache_key: str | None = None) -> dict[str, Any] | None:
        record = self.get_triage_with_meta(arxiv_id_base, cache_key=cache_key)
        return record["data"] if record else None

    def get_triage_with_meta(self, arxiv_id_base: str, cache_key: str | None = None) -> dict[str, Any] | None:
        cache_key = cache_key or arxiv_id_base
        row = self.conn.execute(
            "SELECT triage_json, triage_meta_json FROM triage WHERE arxiv_id_base=?", (cache_key,)
        ).fetchone()
        if not row:
            return None
        return {
            "data": json.loads(row["triage_json"]),
            "meta": json.loads(row["triage_meta_json"]) if row["triage_meta_json"] else None,
        }

    def upsert_triage(
        self,
        month: str,
        triage: dict[str, Any],
        meta: dict[str, Any] | None = None,
        cache_key: str | None = None,
    ) -> None:
        cache_key = cache_key or triage["arxiv_id_base"]
        self.conn.execute(
            """
            INSERT INTO triage(arxiv_id_base, month, triage_json, triage_meta_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(arxiv_id_base) DO UPDATE SET
              month=excluded.month,
              triage_json=excluded.triage_json,
              triage_meta_json=excluded.triage_meta_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (
                cache_key,
                month,
                json.dumps(triage, ensure_ascii=False),
                json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta is not None else None,
            ),
        )
        self.conn.commit()

    def get_summary(self, arxiv_id_base: str, cache_key: str | None = None) -> dict[str, Any] | None:
        record = self.get_summary_with_meta(arxiv_id_base, cache_key=cache_key)
        return record["data"] if record else None

    def get_summary_with_meta(self, arxiv_id_base: str, cache_key: str | None = None) -> dict[str, Any] | None:
        cache_key = cache_key or arxiv_id_base
        row = self.conn.execute(
            "SELECT summary_json, summary_meta_json FROM summaries WHERE arxiv_id_base=?", (cache_key,)
        ).fetchone()
        if not row:
            return None
        return {
            "data": json.loads(row["summary_json"]),
            "meta": json.loads(row["summary_meta_json"]) if row["summary_meta_json"] else None,
        }

    def upsert_summary(
        self,
        month: str,
        summary: dict[str, Any],
        meta: dict[str, Any] | None = None,
        cache_key: str | None = None,
    ) -> None:
        cache_key = cache_key or summary["arxiv_id_base"]
        self.conn.execute(
            """
            INSERT INTO summaries(arxiv_id_base, month, summary_json, summary_meta_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(arxiv_id_base) DO UPDATE SET
              month=excluded.month,
              summary_json=excluded.summary_json,
              summary_meta_json=excluded.summary_meta_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (
                cache_key,
                month,
                json.dumps(summary, ensure_ascii=False),
                json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta is not None else None,
            ),
        )
        self.conn.commit()

    def delete_summary(self, arxiv_id_base: str, cache_key: str | None = None) -> None:
        cache_key = cache_key or arxiv_id_base
        self.conn.execute("DELETE FROM summaries WHERE arxiv_id_base=?", (cache_key,))
        self.conn.commit()

    def upsert_run(self, month: str, stats: dict[str, Any], run_key: str | None = None) -> None:
        run_key = run_key or month
        self.conn.execute(
            """
            INSERT INTO runs(month, stats_json)
            VALUES (?, ?)
            ON CONFLICT(month) DO UPDATE SET
              stats_json=excluded.stats_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (run_key, json.dumps(stats, ensure_ascii=False)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
