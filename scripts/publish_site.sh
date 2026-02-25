#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile_id>" >&2
  exit 1
fi

PROFILE_ID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SRC_DIR="docs/${PROFILE_ID}"
if [[ ! -d "$SRC_DIR" ]]; then
  echo "profile docs not found: $SRC_DIR" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cp -R "$SRC_DIR"/. "$TMP_DIR"/
find docs -mindepth 1 -maxdepth 1 ! -name "$PROFILE_ID" -exec rm -rf {} +
cp -R "$TMP_DIR"/. docs/

echo "Published docs/${PROFILE_ID} -> docs/"
