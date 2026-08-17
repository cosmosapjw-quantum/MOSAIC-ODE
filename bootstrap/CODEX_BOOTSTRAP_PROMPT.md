# MOSAIC-ODE bootstrap prompt for Codex

You are continuing the MOSAIC-ODE integrated ODE solver project from a bootstrap/pre-product repository.

## Mandatory first actions

1. Read `AGENTS.md`.
2. Read `harness/START_HERE.md`, `harness/SCIENTIFIC_CONTRACT.md`, and `harness/VALIDATION_MATRIX.md`.
3. Read `docs/design/MOSAIC_ODE_INTEGRATED_PREPRODUCT_DESIGN.md`.
4. Run `./scripts/verify_preproduct.sh` before editing. If the baseline fails, diagnose the failure before implementation.
5. Read the repository-local skill matching the task under `.agents/skills/`.

## Project constraints

- Mainline is ODE; DAE work is research-only.
- C, C++, CUDA C++, and Python only.
- Residual equations precede scalar losses.
- Topology is core and must affect allocation/branch behavior, but it does not accept steps.
- EESS samples reduced branch/soft variables, not arbitrary full trajectories.
- Online learning must work from current-IVP evidence without requiring pretrained solver weights.
- GPU value is sought through single-IVP internal speculation, device residency, shared RHS/JVP/factorization work, and block/multi-RHS linear algebra.
- Learned/GPU proposals must pass original residual/error validation.
- Do not remove a weak isolated module until the selected A+B interaction experiment has been run.
- Do not claim CUDA speedup without observed CUDA execution and full overhead accounting.

## Immediate development order

1. vector-native residual/JVP contract;
2. repeated-step Robertson/Van der Pol baseline;
3. W=4 window residual/JVP and preconditioners;
4. topology-active 2--4D reduced space;
5. current-IVP vector low-rank preconditioner;
6. device-resident CandidateBundle + fused CUDA residual + H0 primitive;
7. runtime RHS specialization/cache;
8. topology-guided shared-linearization/block-Krylov;
9. bounded conditional CUDA Graph execution.

Use red-green implementation, scientific/numerical validation, reproducibility checks, and independent review before promotion.
