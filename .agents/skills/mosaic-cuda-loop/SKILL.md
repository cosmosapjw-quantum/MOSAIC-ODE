---
name: mosaic-cuda-loop
description: Optimize MOSAIC-ODE CUDA execution for single-IVP internal speculation using device residency, shared work, and profile-driven kernels.
---

# MOSAIC CUDA Loop

1. Preserve an identical CPU/reference semantic path.
2. Prefer device-resident CandidateBundle/EESS/topology/online state over CPU-GPU ping-pong.
3. Fuse RHS/residual/scaling/feature work where profiling supports it.
4. Group topology-compatible candidates by shared linearization and test block/multi-RHS Krylov.
5. Use low precision only for proposals/preconditioners; validate accepted candidates in FP64/original residual.
6. Count launch, transfer, JIT, synchronization, and fallback cost.
7. Move stable bounded workflows into CUDA Graphs only after kernel/state contracts stabilize.
8. Never claim speedup without executable GPU evidence.
