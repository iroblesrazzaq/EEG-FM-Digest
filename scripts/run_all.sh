#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile_id> [config_path]" >&2
  exit 1
fi

PROFILE_ID="$1"
CONFIG_PATH="${2:-configs/batch/all_months.yaml}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python -m eegfm_digest.batch --profile "$PROFILE_ID" --config "$CONFIG_PATH"
