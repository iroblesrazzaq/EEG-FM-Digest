"""Backtest: compare the new merged-query fetch against an existing month
already stored in the production DB.

Read-only against ``data/digest.sqlite`` (opened with ``mode=ro``).  Does
not invoke the pipeline, does not write anything to ``data/`` or
``outputs/``.  Pure HTTP fetch + set comparison.

Usage:
    .venv/bin/python scripts/backtest_merged_query.py 2026-03 [2025-11 ...]
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eegfm_digest.arxiv import fetch_month_candidates  # noqa: E402

DB_PATH = REPO / "data" / "digest.sqlite"


def existing_ids_for_month(month: str) -> set[str]:
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT arxiv_id_base FROM papers WHERE month = ?", (month,)
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def backtest_month(month: str, max_candidates: int = 1000) -> dict:
    print(f"\n=== Backtest {month} ===")
    old_ids = existing_ids_for_month(month)
    print(f"DB has {len(old_ids)} papers for {month}")

    print(f"Fetching with new merged query (max_candidates={max_candidates}) ...")
    new_papers = fetch_month_candidates(
        max_candidates=max_candidates,
        month=month,
        rate_limit_seconds=2.0,
    )
    new_ids = {p["arxiv_id_base"] for p in new_papers}
    print(f"New fetch returned {len(new_ids)} papers for {month}")

    missing = sorted(old_ids - new_ids)
    added = sorted(new_ids - old_ids)
    overlap = old_ids & new_ids

    recall_pct = (100.0 * len(overlap) / len(old_ids)) if old_ids else 0.0
    print(f"Recall: {len(overlap)}/{len(old_ids)} ({recall_pct:.1f}%)")
    print(f"Missing from new fetch (regressions): {len(missing)}")
    for arxiv_id in missing:
        print(f"  - {arxiv_id}")
    print(f"Newly retrieved (added by removing category filter): {len(added)}")
    for arxiv_id in added[:10]:
        title = next(
            (p["title"] for p in new_papers if p["arxiv_id_base"] == arxiv_id),
            "",
        )
        cats = next(
            (",".join(p.get("categories", [])) for p in new_papers if p["arxiv_id_base"] == arxiv_id),
            "",
        )
        print(f"  + {arxiv_id} [{cats}] {title[:80]}")
    if len(added) > 10:
        print(f"  ... and {len(added) - 10} more")

    return {
        "month": month,
        "old": len(old_ids),
        "new": len(new_ids),
        "overlap": len(overlap),
        "missing": len(missing),
        "added": len(added),
        "recall_pct": recall_pct,
    }


def main() -> int:
    months = sys.argv[1:]
    if not months:
        print("usage: backtest_merged_query.py YYYY-MM [YYYY-MM ...]")
        return 2
    results = []
    for i, m in enumerate(months):
        if i > 0:
            time.sleep(1.0)  # polite delay between months
        results.append(backtest_month(m))

    print("\n\n=== Summary ===")
    print(f"{'month':<8}  {'old':>4}  {'new':>4}  {'overlap':>7}  {'missing':>7}  {'added':>5}  {'recall':>7}")
    for r in results:
        print(
            f"{r['month']:<8}  {r['old']:>4}  {r['new']:>4}  {r['overlap']:>7}  "
            f"{r['missing']:>7}  {r['added']:>5}  {r['recall_pct']:>6.1f}%"
        )
    total_old = sum(r["old"] for r in results)
    total_overlap = sum(r["overlap"] for r in results)
    total_missing = sum(r["missing"] for r in results)
    total_added = sum(r["added"] for r in results)
    overall_recall = (100.0 * total_overlap / total_old) if total_old else 0.0
    print(
        f"{'TOTAL':<8}  {total_old:>4}  {sum(r['new'] for r in results):>4}  "
        f"{total_overlap:>7}  {total_missing:>7}  {total_added:>5}  {overall_recall:>6.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
