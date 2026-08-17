# FAILURE_LOG

Known negative result inherited from Integrated V0:

- On the easy scalar BDF1 `y'=y^2` benchmark, the fully integrated path was correct but consumed much more abstract work than the direct backbone.
- This does not invalidate integration; it motivates vector/repeated-step benchmarks, state transport, shared linearization, fused kernels, and device residency.

Known unverified area:

- CUDA execution is not validated in the current environment unless an NVIDIA toolchain/device is later detected.

Resolved bootstrap defects found during independent review:

- wheel metadata initially omitted runtime NumPy/SciPy/PyTorch dependencies; fixed and wheel-smoke verified;
- `git archive` packaging initially allowed a dirty worktree and silently omitted uncommitted source; packaging now refuses dirty state;
- vector EESS population initially persisted rejected-step candidate state and did not transport with predictor drift; transport and rollback semantics are now tested.
