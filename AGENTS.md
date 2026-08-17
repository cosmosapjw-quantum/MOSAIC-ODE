# AGENTS.md — MOSAIC-ODE Integrated Pre-Product

## Mission

MOSAIC-ODE is an integrated ODE solver product-in-development. It combines mature classical numerical methods with topology-aware EESS/homotopy, current-IVP online learning, and GPU internal speculation. The primary goal is integration quality and interaction gain, not novelty inflation in every module.

## Read first

1. `harness/SCIENTIFIC_CONTRACT.md`
2. `harness/VALIDATION_MATRIX.md`
3. `docs/design/MOSAIC_ODE_INTEGRATED_PREPRODUCT_DESIGN.md`
4. `bootstrap/manifest.json`
5. the skill matching the current task under `.agents/skills/`

## Mainline scope

- Mainline: ODEs, including stiff and highly oscillatory/high-frequency systems.
- Experimental: regular mass-matrix systems.
- Research-only: DAEs.
- Implementation languages: C, C++, CUDA C++, Python.

## Core architectural invariants

- The discrete/original residual precedes scalar loss or ML score.
- STEP/WINDOW/FULL share factor definitions.
- topology observes/schedules; homotopy solves paths; degree/endgames audit branches; original residual/error gates accept steps.
- EESS samples reduced branch/soft coordinates, not an unconstrained full trajectory.
- online learning is current-IVP capable and must not require pretrained solver weights.
- GPU value comes from single-IVP internal speculation and shared work, not only external batching.
- low precision may propose or precondition but never bypass FP64/original-equation validation.
- interaction experiments are required before removing a weak isolated module.

## Change policy

Before editing executable behavior:

1. lock the observable outcome;
2. establish a failing test or characterization check;
3. implement the smallest coherent slice;
4. run targeted and regression checks;
5. run scientific/numerical validation where relevant;
6. perform an independent diff review;
7. record unverified GPU/backend areas explicitly.

Do not silently broaden DAE scope, replace original residual validation with ML confidence, or introduce Python callbacks into production hot loops.
