# PLANS.md

## Current task

TASK: Implement the first integrated WeaveODE C/C++/Python vertical slice.

OUTCOME: A buildable package and reproducible benchmark in which classical integration, BDF1/short-window residual solving, topology-aware persistent EESS, homotopy, cold-start online adaptation, and native fused scoring are connected end to end.

IN_SCOPE:
- C++ core and C ABI; CPython/NumPy extension
- DOPRI5, BDF1, Rosenbrock-Euler, exponential midpoint
- W=4 TrajRoot window
- persistent EESS, exact small-cloud VR H0/H1, merge/component tracking
- regular/pseudo-arclength and scalar function-only homotopy
- PyTorch cold-start low-rank online adapter
- CPU native scoring plus optional CUDA source
- tests, scientific validation, reproducibility, synergy benchmark, independent review

OUT_OF_SCOPE:
- DAE implementation
- variable-order production BDF
- complete CTM/KAN/contact atlas
- validated interval numerics
- multi-GPU/MPI
- CUDA performance claim in a CPU-only environment
- packaging to a public index

## Milestones

| ID | Milestone | Files | Validation | Status |
|---|---|---|---|---|
| P-001 | Contracts, build, C/C++ native scoring | CMake/setup/include/cpp | native parity, C smoke | Done |
| P-002 | Classical and structured ODE methods | `src/weaveode/integrators.py` | analytic/order tests | Done |
| P-003 | Root and windowed TrajRoot engines | `nonlinear.py`, `trajroot.py` | linear/nonlinear equivalence | Done |
| P-004 | Topology and persistent EESS | `topology.py`, `eess.py` | H0/H1 and mode retention | Done |
| P-005 | Homotopy engines | `homotopy.py` | fold and function-only telemetry | Done |
| P-006 | Online adapter and integrated solver | `online.py`, `solver.py` | principal branch and rollback | Done |
| P-007 | Benchmarks, reproducibility, review | `benchmarks/`, reports | full matrix and diff review | Done |
