#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 setup.py build_ext --inplace
PYTHONPATH=src:. pytest -q
python3 -m compileall -q src benchmarks
python3 harness/tools/validate_harness.py
cmake -S . -B build-verify -DCMAKE_BUILD_TYPE=Release
cmake --build build-verify --parallel
ctest --test-dir build-verify --output-on-failure
git diff --check
if command -v nvcc >/dev/null 2>&1; then
  echo 'CUDA compiler found; configuring optional CUDA source plane.'
  cmake -S . -B build-cuda-verify -DCMAKE_BUILD_TYPE=Release -DWEAVEODE_ENABLE_CUDA=ON
  cmake --build build-cuda-verify --parallel
else
  echo 'CUDA compiler not found; CUDA execution remains source-only/unverified.'
fi
