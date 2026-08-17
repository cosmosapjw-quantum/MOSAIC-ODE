# Changelog

## 0.2.0a0 — 2026-08-17 — Integrated pre-product bootstrap

- Adopted the MOSAIC-ODE project identity while preserving the validated `weaveode` implementation namespace during bootstrap.
- Added canonical `mosaic_ode` Python facade.
- Added vector BDF1 native residual scoring through C ABI/C++/NumPy extension.
- Added a vector integrated candidate pipeline combining native scoring, EESS/topology, homotopy, online adaptation, and original residual validation.
- Added CPU CandidateBundle reference structure.
- Expanded CUDA source plane for device-resident speculation, topology H0 primitives, low-rank online proposals, multi-vector operations, NVRTC compilation hooks, and graph-control preparation.
- Added integrated harness, repository-local skills, bootstrap manifest, validation scripts, and CI skeleton.
- Kept DAE support research-only and retained all GPU performance claims as unverified until executed on CUDA hardware.
