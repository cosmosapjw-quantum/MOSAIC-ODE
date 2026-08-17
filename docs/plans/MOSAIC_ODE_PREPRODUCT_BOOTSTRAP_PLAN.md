# MOSAIC-ODE Integrated pre-Product Bootstrap Plan

## Goal

Seed a GitHub-ready codebase that preserves the verified Integrated V0 behavior while making the next product-development contracts explicit.

## Completed bootstrap slices

- canonical MOSAIC-ODE identity with legacy `weaveode` namespace compatibility;
- C ABI/C++20 vector residual plane;
- scalar and vector integrated pipelines;
- persistent EESS, topology and homotopy participation;
- current-IVP cold-start online adaptation;
- CUDA source plane for device-resident candidates, fused vector scoring, H0 primitives, low-rank proposals, multi-vector operations and NVRTC hooks;
- research/coding harness and repository-local skills;
- CI, packaging and verification scripts.

## Next implementation order

1. repeated vector implicit steps on Robertson/Van der Pol;
2. W=4 vector all-at-once residual/JVP and preconditioners;
3. topology-active reduced candidate state;
4. current-IVP vector low-rank preconditioning;
5. executable CUDA CandidateBundle and fused scoring;
6. topology-guided shared-linearization/block-Krylov;
7. runtime RHS specialization/cache;
8. conditional CUDA Graph speculative epoch.

## Promotion rule

No performance or GPU claim is promoted until the relevant executable backend is observed at equal original-space error with all adaptation/JIT/launch/transfer/synchronization/fallback costs included.
