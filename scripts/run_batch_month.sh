#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 YYYY-MM" >&2
  exit 1
fi

MONTH="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TMP_CONFIG="$(mktemp)"
trap 'rm -f "$TMP_CONFIG"' EXIT

python - "$MONTH" <<'PY' > "$TMP_CONFIG"
import json
import sys
from pathlib import Path

month = sys.argv[1]
cfg = json.loads(Path("configs/batch_single_month.json").read_text(encoding="utf-8"))
cfg["months"] = [month]
print(json.dumps(cfg, ensure_ascii=False))
PY

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python -m eegfm_digest.batch --config "$TMP_CONFIG"
