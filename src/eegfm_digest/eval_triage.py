from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("data/digest.sqlite")
DEFAULT_GOLD_PATH = Path("tests/fixtures/triage_eval_gold_v1.jsonl")
VALID_DECISIONS = {"accept", "reject", "borderline"}
VALID_GROUPED = {"accept", "not_pass"}


def normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def group_decision(decision: str) -> str:
    normalized = decision.strip().lower()
    if normalized == "accept":
        return "accept"
    if normalized in {"reject", "borderline"}:
        return "not_pass"
    raise ValueError(f"Unsupported triage decision: {decision!r}")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _read_titles(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise RuntimeError(f"JSONL row is not an object in {path}")
        rows.append(data)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _query_join_row_by_title_like(conn: sqlite3.Connection, title: str) -> list[dict[str, str]]:
    escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    query = """
        SELECT
          p.month AS month,
          p.arxiv_id_base AS arxiv_id_base,
          json_extract(p.metadata_json, '$.arxiv_id') AS arxiv_id,
          json_extract(p.metadata_json, '$.title') AS title,
          json_extract(t.triage_json, '$.decision') AS decision
        FROM papers p
        JOIN triage t ON t.arxiv_id_base = p.arxiv_id_base
        WHERE lower(json_extract(p.metadata_json, '$.title')) LIKE lower(?) ESCAPE '\\'
        ORDER BY p.month DESC, p.arxiv_id_base ASC
    """
    out: list[dict[str, str]] = []
    for row in conn.execute(query, (pattern,)):
        decision = str(row["decision"]).strip().lower()
        if decision not in VALID_DECISIONS:
            continue
        out.append(
            {
                "month": str(row["month"]),
                "arxiv_id_base": str(row["arxiv_id_base"]),
                "arxiv_id": str(row["arxiv_id"]),
                "title": str(row["title"]),
                "decision": decision,
            }
        )
    return out


def _query_all_join_rows(conn: sqlite3.Connection) -> list[dict[str, str]]:
    query = """
        SELECT
          p.month AS month,
          p.arxiv_id_base AS arxiv_id_base,
          json_extract(p.metadata_json, '$.arxiv_id') AS arxiv_id,
          json_extract(p.metadata_json, '$.title') AS title,
          json_extract(t.triage_json, '$.decision') AS decision
        FROM papers p
        JOIN triage t ON t.arxiv_id_base = p.arxiv_id_base
        ORDER BY p.month DESC, p.arxiv_id_base ASC
    """
    rows: list[dict[str, str]] = []
    for row in conn.execute(query):
        decision = str(row["decision"]).strip().lower()
        if decision not in VALID_DECISIONS:
            continue
        rows.append(
            {
                "month": str(row["month"]),
                "arxiv_id_base": str(row["arxiv_id_base"]),
                "arxiv_id": str(row["arxiv_id"]),
                "title": str(row["title"]),
                "decision": decision,
            }
        )
    return rows


def _resolve_title(
    conn: sqlite3.Connection,
    title: str,
    normalized_index: dict[str, list[dict[str, str]]],
) -> dict[str, str]:
    normalized = normalize_title(title)
    exact_rows = normalized_index.get(normalized, [])
    if len(exact_rows) == 1:
        return exact_rows[0]
    if len(exact_rows) > 1:
        ids = ", ".join(sorted(row["arxiv_id_base"] for row in exact_rows))
        raise RuntimeError(f"Ambiguous exact title match for {title!r}: {ids}")

    fallback_rows = _query_join_row_by_title_like(conn, re.sub(r"\s+", " ", title.strip()))
    if len(fallback_rows) == 1:
        return fallback_rows[0]
    if len(fallback_rows) > 1:
        ids = ", ".join(sorted(row["arxiv_id_base"] for row in fallback_rows))
        raise RuntimeError(f"Ambiguous fallback title match for {title!r}: {ids}")
    raise RuntimeError(f"Title not found in papers+triage join: {title!r}")


def build_gold_snapshot(db_path: Path, titles_path: Path, out_path: Path) -> list[dict[str, str]]:
    conn = _connect(db_path)
    try:
        all_rows = _query_all_join_rows(conn)
        normalized_index: dict[str, list[dict[str, str]]] = {}
        for row in all_rows:
            normalized_index.setdefault(normalize_title(row["title"]), []).append(row)

        seen_ids: set[str] = set()
        snapshot_rows: list[dict[str, str]] = []
        for title in _read_titles(titles_path):
            matched = _resolve_title(conn, title, normalized_index)
            arxiv_id_base = matched["arxiv_id_base"]
            if arxiv_id_base in seen_ids:
                raise RuntimeError(f"Duplicate arxiv_id_base resolved from title list: {arxiv_id_base}")
            seen_ids.add(arxiv_id_base)
            source_decision = matched["decision"]
            snapshot_rows.append(
                {
                    "arxiv_id_base": arxiv_id_base,
                    "arxiv_id": matched["arxiv_id"],
                    "title": matched["title"],
                    "month": matched["month"],
                    "gold_grouped": group_decision(source_decision),
                    "source_decision": source_decision,
                    "notes": "",
                }
            )
    finally:
        conn.close()

    snapshot_rows = sorted(snapshot_rows, key=lambda row: row["arxiv_id_base"])
    _write_jsonl(out_path, snapshot_rows)
    return snapshot_rows


def _query_decision_map(conn: sqlite3.Connection, arxiv_ids: list[str]) -> dict[str, str]:
    if not arxiv_ids:
        return {}
    placeholders = ",".join("?" for _ in arxiv_ids)
    query = f"""
        SELECT
          arxiv_id_base,
          json_extract(triage_json, '$.decision') AS decision
        FROM triage
        WHERE arxiv_id_base IN ({placeholders})
    """
    out: dict[str, str] = {}
    for row in conn.execute(query, arxiv_ids):
        decision = str(row["decision"]).strip().lower()
        if decision in VALID_DECISIONS:
            out[str(row["arxiv_id_base"])] = decision
    return out


def compute_confusion(pairs: list[tuple[str, str]]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for gold, pred in pairs:
        if gold == "accept":
            if pred == "accept":
                counts["tp"] += 1
            else:
                counts["fn"] += 1
        else:
            if pred == "accept":
                counts["fp"] += 1
            else:
                counts["tn"] += 1
    counts["total"] = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    return counts


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def score_gold_snapshot(db_path: Path, gold_path: Path) -> dict[str, Any]:
    gold_rows = _load_jsonl(gold_path)
    required = {
        "arxiv_id_base",
        "arxiv_id",
        "title",
        "month",
        "gold_grouped",
        "source_decision",
        "notes",
    }
    for row in gold_rows:
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"Gold row missing keys {missing}: {row}")
        gold_grouped = str(row["gold_grouped"])
        if gold_grouped not in VALID_GROUPED:
            raise RuntimeError(f"Invalid gold_grouped={gold_grouped!r} in row {row}")

    arxiv_ids = [str(row["arxiv_id_base"]) for row in gold_rows]
    conn = _connect(db_path)
    try:
        decision_map = _query_decision_map(conn, arxiv_ids)
    finally:
        conn.close()

    missing_ids: list[str] = []
    scored_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = []

    for row in gold_rows:
        arxiv_id_base = str(row["arxiv_id_base"])
        pred_decision = decision_map.get(arxiv_id_base)
        if pred_decision is None:
            missing_ids.append(arxiv_id_base)
            continue
        gold_grouped = str(row["gold_grouped"])
        pred_grouped = group_decision(pred_decision)
        correct = gold_grouped == pred_grouped
        scored = {
            "arxiv_id_base": arxiv_id_base,
            "title": str(row["title"]),
            "gold_grouped": gold_grouped,
            "pred_grouped": pred_grouped,
            "pred_decision": pred_decision,
            "correct": correct,
        }
        scored_rows.append(scored)
        pairs.append((gold_grouped, pred_grouped))
        if not correct:
            mismatch_rows.append(scored)

    confusion = compute_confusion(pairs)
    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    tn = confusion["tn"]
    total = confusion["total"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    accuracy = _safe_div(tp + tn, total)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    metrics = {
        "accuracy": accuracy,
        "precision_accept": precision,
        "recall_accept": recall,
        "f1_accept": f1,
        "support_accept": tp + fn,
        "support_not_pass": tn + fp,
        "pred_accept": tp + fp,
        "pred_not_pass": tn + fn,
    }

    return {
        "gold_total": len(gold_rows),
        "scored_total": len(scored_rows),
        "missing_ids": sorted(missing_ids),
        "confusion": confusion,
        "metrics": metrics,
        "mismatches": mismatch_rows,
    }


def _print_score_report(report: dict[str, Any]) -> None:
    confusion = report["confusion"]
    metrics = report["metrics"]
    print(f"Gold rows: {report['gold_total']}")
    print(f"Scored rows: {report['scored_total']}")
    print(f"Missing IDs: {len(report['missing_ids'])}")
    if report["missing_ids"]:
        for arxiv_id_base in report["missing_ids"]:
            print(f"  - {arxiv_id_base}")
    print("")
    print("Confusion Matrix (positive class: accept)")
    print("                pred_accept  pred_not_pass")
    print(f"gold_accept     {confusion['tp']:11d}  {confusion['fn']:13d}")
    print(f"gold_not_pass   {confusion['fp']:11d}  {confusion['tn']:13d}")
    print("")
    print("Metrics")
    print(f"  accuracy:         {metrics['accuracy']:.4f}")
    print(f"  precision_accept: {metrics['precision_accept']:.4f}")
    print(f"  recall_accept:    {metrics['recall_accept']:.4f}")
    print(f"  f1_accept:        {metrics['f1_accept']:.4f}")
    print(f"  support_accept:   {metrics['support_accept']}")
    print(f"  support_not_pass: {metrics['support_not_pass']}")
    if report["mismatches"]:
        print("")
        print("Mismatches")
        print("arxiv_id_base\tgold_grouped\tpred_grouped\tpred_decision\ttitle")
        for row in sorted(report["mismatches"], key=lambda r: r["arxiv_id_base"]):
            title = row["title"].replace("\t", " ").replace("\n", " ")
            print(
                f"{row['arxiv_id_base']}\t{row['gold_grouped']}\t{row['pred_grouped']}\t"
                f"{row['pred_decision']}\t{title}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual triage evaluation harness (DB-backed) with accept vs not_pass scoring."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_gold = sub.add_parser(
        "build-gold",
        help="Build or refresh gold snapshot JSONL from a title list and current triage DB decisions.",
    )
    build_gold.add_argument("--titles-file", required=True, help="Path to title list text file.")
    build_gold.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to sqlite DB.")
    build_gold.add_argument("--out", default=str(DEFAULT_GOLD_PATH), help="Output JSONL path.")

    score = sub.add_parser("score", help="Score gold snapshot against current triage decisions in sqlite DB.")
    score.add_argument("--gold", default=str(DEFAULT_GOLD_PATH), help="Gold JSONL path.")
    score.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to sqlite DB.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-gold":
        try:
            rows = build_gold_snapshot(
                db_path=Path(args.db),
                titles_path=Path(args.titles_file),
                out_path=Path(args.out),
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote gold snapshot: {args.out}")
        print(f"Rows: {len(rows)}")
        return 0

    if args.command == "score":
        try:
            report = score_gold_snapshot(db_path=Path(args.db), gold_path=Path(args.gold))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_score_report(report)
        if report["missing_ids"]:
            return 2
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
