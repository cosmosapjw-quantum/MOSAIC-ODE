# DECISION_LOG

- 2026-08-17: project-facing name set to MOSAIC-ODE; `weaveode` retained as legacy implementation namespace during bootstrap.
- 2026-08-17: ODE mainline, DAE research-only.
- 2026-08-17: topology, EESS/homotopy, current-IVP online learning, and GPU internal speculation remain core architecture.
- 2026-08-17: C/C++/CUDA C++/Python are the implementation languages for this bootstrap.
- 2026-08-17: device residency, topology-guided shared-linearization groups, block Krylov, runtime RHS specialization, and CUDA Graph control are promoted to explicit GPU research targets.
- 2026-08-17: device/CUDA source plane is included as bootstrap code, but CUDA execution remains an explicit proof gap until nvcc/device validation exists.
- 2026-08-17: packaging requires a clean committed worktree so the source ZIP cannot silently omit uncommitted bootstrap files.
- 2026-08-17: vector EESS transport is predictor-centered across accepted steps and is rolled back on rejected steps.
