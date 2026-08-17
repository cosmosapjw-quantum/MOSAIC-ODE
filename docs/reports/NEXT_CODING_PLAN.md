# MOSAIC-ODE Next Coding Plan

## Objective

Move the integrated pre-product from scalar/one-step vector proof toward repeated vector implicit integration where topology, EESS, homotopy, online adaptation and GPU shared work can create measurable interaction gain.

## Primary problems

1. Robertson kinetics with positivity and mass audits.
2. Van der Pol at mu=1e2 and 1e3 with stiffness-transition windows.
3. Chirped two-mode oscillator for phase and H1/winding allocation.

## Work packages

### V1.1 vector repeated-step contract
Generalize RHS/JVP/candidate telemetry and establish repeated BDF/Rosenbrock baselines.

### V1.2 W=4 vector window solve
Add global residual/JVP, block-triangular, block-Jacobi and overlap-Schwarz preconditioners.

### V1.3 topology-active reduced state
Build 2--4D branch/soft representations and require topology to alter path/EESS allocation.

### V1.4 current-IVP vector preconditioner
Compare bounded low-rank adaptation against extrapolation, Broyden, Anderson, diagonal scaling and reused Jacobians/preconditioners with adaptation cost included.

### V1.5 executable CUDA CandidateBundle
Add device memory pool, SoA candidates, fused vector residual scoring, compaction and H0 component primitives while preserving CPU parity.

### V1.6 runtime RHS specialization
JIT a restricted ODE expression contract to CUDA, cache artifacts, and verify FP64 parity with the CPU expression backend.

### V1.7 topology-guided block Krylov
Group related candidate/path RHS by topology/branch/linearization similarity and compare K independent Krylov solves against a shared block/multi-RHS solve.

### V1.8 CUDA Graph speculative epoch
After kernel/state contracts stabilize, capture bounded DIRECT/JFNK/EESS/HOMOTOPY/VALIDATE/UPDATE workflows to reduce host round trips.

## Gate

All comparisons use equal original-space global error and include RHS/JVP/J work, nonlinear iterations, branch robustness, topology effects, training/adaptation, JIT, launch, transfer, synchronization, fallback and memory. A weak isolated module is not removed before selected interaction configurations have been measured.
