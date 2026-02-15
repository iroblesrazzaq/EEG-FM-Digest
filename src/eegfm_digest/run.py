from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date

from dateutil.relativedelta import relativedelta

from .config import load_config
from .pipeline import run_month


def default_month() -> str:
    first = date.today().replace(day=1)
    prev = first - relativedelta(months=1)
    return f"{prev.year:04d}-{prev.month:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default=default_month(), help="YYYY-MM (default: previous month)")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-accepted", type=int, default=None)
    parser.add_argument("--include-borderline", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if args.max_candidates is not None:
        cfg = replace(cfg, max_candidates=args.max_candidates)
    if args.max_accepted is not None:
        cfg = replace(cfg, max_accepted=args.max_accepted)
    if args.include_borderline:
        cfg = replace(cfg, include_borderline=True)

    run_month(cfg, args.month, no_pdf=args.no_pdf, force=args.force)


if __name__ == "__main__":
    main()
