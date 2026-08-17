---
name: mosaic-independent-review
description: Review MOSAIC-ODE changes independently for numerical correctness, scientific contract violations, GPU claim inflation, regressions, and missing interaction tests.
---

# Independent Review

1. Review the spec/contract before the diff.
2. Trace CPU, GPU, topology, EESS, homotopy, and online-learning runtime paths.
3. Prioritize false acceptance, branch errors, derivative-mode violations, numerical instability, unsafe memory/FFI, and silent fallbacks.
4. Check that performance claims include all overhead and that unexecuted CUDA is labeled unverified.
5. Check interaction experiments before recommending deletion of an isolated weak module.
6. Report severity, evidence, impact, required tests, and residual risks; do not silently fix during review.
