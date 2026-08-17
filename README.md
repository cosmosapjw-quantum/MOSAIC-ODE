# MOSAIC-ODE Integrated pre-Product

MOSAIC-ODE is a **GPU-assisted, topology-aware, online-adaptive integrated ODE solver** in pre-product bootstrap form.

The project is deliberately integration-first: it combines mature classical numerical methods with a residual-first multi-scope formulation, topology-aware EESS, homotopy/branch tracking, current-IVP online learning, and single-IVP GPU internal speculation. The experimental modules are not allowed to bypass the original numerical residual/error gate.

This repository descends from the verified WeaveODE Integrated V0 coding loop. The implementation namespace `weaveode` remains temporarily available for compatibility, while new Python code should import `mosaic_ode`.

## What is already executable

- C ABI and C++20 native primitives.
- CPython/NumPy native extension.
- Adaptive Dormand–Prince 5(4), BDF1, Rosenbrock–Euler, exponential midpoint.
- Explicit-J, finite-difference, and matrix-free JVP Newton paths.
- Short all-at-once BDF1 window solve.
- Persistent finite EESS state.
- H0/H1 small-cloud topology and merge profiles.
- Regular continuation, pseudo-arclength fold traversal, strict function-only scalar continuation.
- Cold-start current-IVP low-rank online adapter.
- Scalar integrated BDF1 vertical slice.
- Vector-native BDF1 candidate scoring and vector integrated candidate/homotopy pipeline.
- CUDA C++ source plane for device-resident candidate bundles, fused vector scoring, H0 label propagation, low-rank proposals, multi-vector primitives, runtime NVRTC hooks, and future graph control.

## What is not claimed yet

- No validated CUDA execution or GPU speedup in the bootstrap environment.
- No production-grade universal variable-order BDF implementation.
- No calibrated Bayesian uncertainty claim.
- No broad DAE production support; DAE work is research-only.
- No claim that persistent homology implies homotopy equivalence.

## Quick start

```bash
./scripts/bootstrap.sh
./scripts/verify_preproduct.sh
```

or manually:

```bash
python3 setup.py build_ext --inplace
PYTHONPATH=src:. pytest -q
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Minimal canonical import:

```python
import mosaic_ode
print(mosaic_ode.__project_name__)
```

## Repository map

- `src/weaveode/`: validated implementation lineage.
- `src/mosaic_ode/`: canonical pre-product Python facade.
- `include/weaveode/`, `cpp/`: C ABI and C++ numerical core.
- `cuda/`: optional CUDA C++ device plane.
- `harness/`: integrated research/coding workflow and state.
- `.agents/skills/`: repository-local agent skills.
- `docs/design/`: canonical product/numerical design.
- `docs/architecture/`: focused subsystem contracts.
- `bootstrap/`: GitHub/bootstrap metadata.

## Development principles

1. Residual equations precede scalar losses.
2. ODE is mainline; DAE is research scope.
3. Topology is core but does not accept steps.
4. Online learning must work from current-IVP evidence without requiring pretrained solver weights.
5. GPU value is sought through internal single-IVP candidate/path parallelism and shared work.
6. Weak isolated modules are not removed before selected interaction experiments are completed.
7. Performance claims include training/adaptation, JIT, launch, transfer, synchronization, and fallback cost.

Read `AGENTS.md` and `docs/design/MOSAIC_ODE_INTEGRATED_PREPRODUCT_DESIGN.md` before substantial development.
