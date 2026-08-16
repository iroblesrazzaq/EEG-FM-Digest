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
        record = self.get_triage_with_meta(arxiv_id_base)
        return record["data"] if record else None

    def get_triage_with_meta(self, arxiv_id_base: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT triage_json, triage_meta_json FROM triage WHERE arxiv_id_base=?", (arxiv_id_base,)
        ).fetchone()
        if not row:
            return None
        return {
            "data": json.loads(row["triage_json"]),
            "meta": json.loads(row["triage_meta_json"]) if row["triage_meta_json"] else None,
        }

    def upsert_triage(self, month: str, triage: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
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
                triage["arxiv_id_base"],
                month,
                json.dumps(triage, ensure_ascii=False),
                json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta is not None else None,
            ),
        )
        self.conn.commit()

    def get_summary(self, arxiv_id_base: str) -> dict[str, Any] | None:
        record = self.get_summary_with_meta(arxiv_id_base)
        return record["data"] if record else None

    def get_summary_with_meta(self, arxiv_id_base: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT summary_json, summary_meta_json FROM summaries WHERE arxiv_id_base=?", (arxiv_id_base,)
        ).fetchone()
        if not row:
            return None
        return {
            "data": json.loads(row["summary_json"]),
            "meta": json.loads(row["summary_meta_json"]) if row["summary_meta_json"] else None,
        }

    def upsert_summary(self, month: str, summary: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
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
                summary["arxiv_id_base"],
                month,
                json.dumps(summary, ensure_ascii=False),
                json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta is not None else None,
            ),
        )
        self.conn.commit()

    def delete_summary(self, arxiv_id_base: str) -> None:
        self.conn.execute("DELETE FROM summaries WHERE arxiv_id_base=?", (arxiv_id_base,))
        self.conn.commit()

    def get_paper(self, arxiv_id_base: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT metadata_json FROM papers WHERE arxiv_id_base=?",
            (arxiv_id_base,),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["metadata_json"])

    def list_papers_for_month(self, month: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT metadata_json FROM papers WHERE month=? ORDER BY arxiv_id_base",
            (month,),
        ).fetchall()
        return [json.loads(row["metadata_json"]) for row in rows]

    def list_triage_for_month(self, month: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT triage_json FROM triage WHERE month=? ORDER BY arxiv_id_base",
            (month,),
        ).fetchall()
        return [json.loads(row["triage_json"]) for row in rows]

    def list_summaries_for_month(self, month: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT summary_json FROM summaries WHERE month=? ORDER BY arxiv_id_base",
            (month,),
        ).fetchall()
        return [json.loads(row["summary_json"]) for row in rows]

    def get_accepted_without_summary(self) -> list[tuple[str, str]]:
        """Return ``(month, arxiv_id_base)`` for accepts that need a full-text summary.

        Includes accepts with no summary row, and accepts whose stored summary is
        abstract-only (``used_fulltext`` is not true) so a later PDF success can
        upgrade them.
        """
        rows = self.conn.execute(
            """
            SELECT t.month, t.arxiv_id_base, t.triage_json, s.summary_json
            FROM triage t
            LEFT JOIN summaries s ON s.arxiv_id_base = t.arxiv_id_base
            WHERE s.arxiv_id_base IS NULL
               OR IFNULL(json_extract(s.summary_json, '$.used_fulltext'), 0) != 1
            ORDER BY t.month, t.arxiv_id_base
            """
        ).fetchall()
        out: list[tuple[str, str]] = []
        for row in rows:
            try:
                data = json.loads(row["triage_json"])
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("decision") != "accept":
                continue
            out.append((str(row["month"]), str(row["arxiv_id_base"])))
        return out

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
