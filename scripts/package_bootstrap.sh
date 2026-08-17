#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "Refusing to package a dirty worktree: commit or clean tracked/untracked source first." >&2
  exit 2
fi
OUT="${1:-artifacts/bootstrap}"
mkdir -p "$OUT"
NAME="MOSAIC-ODE_integrated_preproduct_$(date +%Y%m%d)"
git archive --format=zip --output="$OUT/${NAME}.zip" HEAD
git bundle create "$OUT/${NAME}.bundle" --all
sha256sum "$OUT/${NAME}.zip" "$OUT/${NAME}.bundle" > "$OUT/${NAME}.sha256"
echo "$OUT/${NAME}.zip"
echo "$OUT/${NAME}.bundle"
