#!/usr/bin/env bash
# Fetch CFA (Naive Concat/Add 所在仓) into third_party/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/cfa"
URL="${CFA_URL:-https://github.com/seunghan96/cfa.git}"

if [[ -d "$DEST/.git" ]]; then
  echo "Already present: $DEST"
  git -C "$DEST" remote -v
  git -C "$DEST" log -1 --oneline
  exit 0
fi

mkdir -p "$ROOT/third_party"
git clone --depth 1 "$URL" "$DEST"
echo "Cloned CFA -> $DEST"
git -C "$DEST" log -1 --oneline
