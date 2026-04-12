#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from eegfm_digest.csv_export import export_all_csv


def main() -> None:
    export_all_csv(ROOT_DIR / "outputs")


if __name__ == "__main__":
    main()
