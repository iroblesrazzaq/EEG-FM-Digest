#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <profile_id> <YYYY-MM>" >&2
  exit 1
fi

PROFILE_ID="$1"
MONTH="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python -m eegfm_digest.run --profile "$PROFILE_ID" --month "$MONTH"
