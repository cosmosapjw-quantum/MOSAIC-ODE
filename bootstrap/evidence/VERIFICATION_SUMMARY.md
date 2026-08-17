# MOSAIC-ODE pre-product verification summary

Date: 2026-08-17
Stage: integrated pre-product bootstrap

Observed local proof:

- Python/extension build: pass.
- pytest: **41 passed**.
- CMake/CTest: **2/2 passed**.
- ASan/UBSan C/C++ smoke: **2/2 passed**.
- Harness validator: pass.
- Python compileall: pass.
- `git diff --check`: pass.
- Focused Python line coverage: **86%**.
- Wheel build/install smoke: pass.
- Wheel runtime dependency metadata: `numpy>=1.26`, `scipy>=1.11`, `torch>=2.2`.
- Inherited scalar integrated benchmark: executable, correctness retained.
- Vector pre-product benchmark: executable; linear BDF1 discrete-root parity and Robertson positivity/mass audit pass in the tested case.
- CUDA: **source-only/unverified** because no CUDA compiler/device is available in this validation environment.

No GPU performance claim is made by this package.
