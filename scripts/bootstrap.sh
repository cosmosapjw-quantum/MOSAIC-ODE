#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 setup.py build_ext --inplace
python3 harness/tools/validate_harness.py
printf 'MOSAIC-ODE bootstrap complete. Run ./scripts/verify_preproduct.sh for the full proof gate.\n'
