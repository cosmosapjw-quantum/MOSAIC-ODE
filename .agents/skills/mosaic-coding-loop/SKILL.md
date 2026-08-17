---
name: mosaic-coding-loop
description: Implement MOSAIC-ODE research code with red-green proof, scientific validation, reproducibility, and independent review.
---

# MOSAIC Coding Loop

1. Read AGENTS.md, SCIENTIFIC_CONTRACT.md, VALIDATION_MATRIX.md, and the relevant design.
2. Lock one observable slice and protected behavior.
3. Reproduce baseline and write/identify a failing check before code.
4. Implement the smallest coherent C/C++/CUDA/Python change.
5. Run targeted tests, regression, numerical validation, and sanitizer/build checks.
6. Record work accounting including GPU/JIT/adaptation overhead.
7. Run independent diff review before promotion.
8. Update run/decision/failure state.
