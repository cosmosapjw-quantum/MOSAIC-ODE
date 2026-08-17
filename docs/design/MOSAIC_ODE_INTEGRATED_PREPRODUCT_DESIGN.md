# MOSAIC-ODE Integrated Product Design v1

**Status:** canonical integrated pre-product bootstrap baseline, 2026-08-17  
**Languages:** C, C++, CUDA C++, Python  
**Mainline:** ODEs, including stiff and highly oscillatory/high-frequency systems  
**Research scope:** mass-matrix systems and DAEs

## Definition

MOSAIC-ODE is a residual-first, multi-scope ODE solver in which classical time integrators provide deterministic solving and validation, while topology-aware EESS, homotopy/branch machinery, current-IVP online learning, and GPU internal speculation actively generate and improve candidate steps and nonlinear roots.

The project is integration-first rather than a claim that every component is a new algorithm. The target is interaction gain between strong existing methods and experimental modules.

## Product intent

Core architectural components are present from the start but conditionally active:

- classical explicit/stiff/oscillatory method portfolio;
- STEP/WINDOW/FULL residual scopes sharing the same factors;
- J-explicit, matrix-free Jv, and strict function-only root lanes;
- topology observer and branch graph;
- persistent reduced-space EESS;
- homotopy, pseudo-arclength, singular-endpoint/endgame machinery;
- current-IVP online learning without a pretrained-solver requirement;
- single-IVP GPU internal speculation;
- original-equation residual/error validation and deterministic fallback.

A weak isolated marginal result does not remove a module until selected A+B interaction experiments are run.

## Residual-first core

For local factors

\[
r_n(u_n;h_n)=0,
\]

window/global solves use

\[
R_{a:b}(z)=[r_a,\ldots,r_{b-1}]^T=0.
\]

The hierarchy is

\[
\boxed{\text{discrete equation}>\text{residual metric}>\text{solver/optimizer}}.
\]

L-BFGS, Newton, JFNK, homotopy, EESS residual measures, and learned proposals are different computational routes to the same feasibility object; a small scalar loss does not replace the primal residual.

## Execution scopes

- **STEP:** default sequential solve.
- **WINDOW:** short all-at-once solve, initially W in {2,4,8}, promoted under stagnation, rejection clusters, stiffness transitions, branch ambiguity, phase/invariant drift, topology events, or a positive cost prediction.
- **FULL:** supported but not default; reserved for global branch/periodic tasks, long-range constraints, time-parallel/global-inference research, and regimes where memory/cost gates permit it.

## Classical portfolio

- Nonstiff: Dormand-Prince/Tsitouras-class embedded RK, dense output, event handling.
- Stiff: Rosenbrock-W and BDF with Newton/JFNK.
- Oscillatory/high-frequency: commutator-free Magnus/exponential-Krylov, optional symplectic/interaction-picture expert.
- Mature SUNDIALS/PETSc methods may be used as production adapters/baselines rather than duplicated.

## Nonlinear and homotopy taxonomy

Dispatch is defined on three independent axes:

```text
scope:       STEP / WINDOW / FULL
derivatives: EXPLICIT_J / MATRIX_FREE_JV / FUNCTION_ONLY
geometry:    REGULAR / SIMPLE_FOLD / SINGULAR_ENDPOINT / NONSMOOTH_OR_NOISY
```

Ordinary methods are attempted before expensive escalation: damped Newton/trust region, JFNK, Broyden/multisecant, Anderson, pseudo-transient continuation, pseudo-arclength, deflation/branch switching, reduced function-only homotopy, then EESS/learned homotopy.

A simple fold may have singular H_x but regular augmented [H_x,H_lambda] and is handled by pseudo-arclength. A genuinely singular endpoint is not repaired by an invertible coordinate transform; the path is singular-subspace estimation -> Lyapunov-Schmidt reduction -> EESS branch hypotheses -> Jv/function-only homotopy -> endgame/branch switching.

## Topology, homotopy and EESS

Roles are fixed:

- homology/topology: observer and scheduler;
- EESS: reduced branch/soft hypothesis population;
- homotopy: zero-path solver;
- degree/index/endgame: branch audit;
- original residual/error gate: numerical acceptance.

Persistent H0/H1 and Reeb/merge structure may alter EESS representatives, homotopy path count, branch-switch seeds, window size, and shared-linearization groups, but do not certify a numerical step.

EESS never samples an unconstrained full trajectory. A typical variable is

\[
\xi=(a,b,\theta_H,m),
\]

with low-dimensional branch/soft coordinates a, branch identity b, homotopy parameters and solver metadata. A Gaussian zero-residual pseudo-observation may define

\[
p(0|\xi)\propto\exp[-\tfrac12 r(\xi)^T\Sigma^{-1}r(\xi)],
\]

otherwise generalized-Bayes/Gibbs terminology is used. Compute cost is controller utility, not likelihood.

## Online learning

A new IVP can cold-start from neutral/zero learned state. Pretraining is optional only. Current-IVP telemetry includes state/derivative/residual changes, timestep, nonlinear/Krylov history, and topology state.

Online outputs may include bounded trajectory/root corrections, low-rank preconditioner updates, local chart/normal-form parameters, endpoint-preserving homotopy deformation, and EESS/path/window budgets. Every learned output must pass the original residual/error gate.

## GPU architecture

The GPU is a single-IVP internal co-solver, not merely an external batch accelerator. Device-resident speculative state should include candidate bundles, EESS population and ancestry, topology/branch metadata, Krylov/multi-RHS workspace, and compact online-model state.

Target fast path:

```text
predictor
 -> fused RHS/residual/constraints/features
 -> topology + EESS update
 -> DIRECT/JFNK/EESS/HOMOTOPY branch
 -> candidate/path correction
 -> FP64 original-residual validation
 -> online update
```

Key interaction target:

\[
\boxed{\text{topology clustering}\to\text{shared-linearization groups}\to\text{block/multi-RHS Krylov}\to\text{CUDA Graph execution}}.
\]

CUDA work includes fused vector residual scoring, multi-JVP/block operations, H0/merge primitives, batched small bordered/reduced solves, low-rank online updates, device memory pools, runtime RHS specialization via NVRTC, and later conditional CUDA Graph control. cuSPARSE/cuSolverDx/cuDSS are optional acceleration backends where their mathematical assumptions match.

Precision policy:

\[
\text{FP32/TF32 proposals or preconditioners}\to\text{FP32/FP64 corrections}\to\text{FP64 original-residual validation}.
\]

No GPU speed claim is made until real CUDA execution includes launch, transfer, JIT, synchronization, adaptation and fallback costs.

## Implementation ownership

- C++20/23: numerical state, integrators, residual/root engines, production topology/EESS orchestration, online runtime, GPU dispatch, adapters, validation and telemetry.
- CUDA C++: hot device kernels and device-resident speculative workflow.
- C ABI: stable public native boundary.
- Python: user/research API, configuration, prototype systems, reporting and benchmark orchestration; Python stays out of production hot loops.

## Validation semantics

Candidate acceptance requires the original scaled residual, independent local/discretization error information, hard constraints and branch continuity, plus phase/structure checks when applicable. Derivative-access telemetry records J, JVP/VJP, finite-difference derivative and function-only calls. `FUNCTION_ONLY` requires zero derivative calls.

CPU and CUDA accelerated residuals must share the same mathematical expression/contract and be parity-tested in FP64. CPU remains an independent audit/fallback path.

## Interaction experiment

Modules: B classical backbone, W window solve, E EESS, T topology, H homotopy, M online learning, G GPU speculation, A geometric atlas/normal form.

Mandatory configurations include isolated and selected interactions such as B+T+E, B+T+H, B+E+H, B+W+M, B+G+E+T+H, and the integrated B+G+E+T+H+M+W. Compare at equal original-space global error and report wall time, RHS/JVP/J work, nonlinear iterations, branch recall, false acceptance, phase/constraint error, GPU overhead, adaptation cost and memory.

## Current status and next cycle

Integrated V0 already connected classical prediction/correction, C++ scoring, persistent EESS, topology-aware components, BDF1 homotopy, cold-start current-IVP low-rank adaptation, principal-branch selection and FP64 residual validation. Its easy scalar benchmark was correct but much more expensive than direct BDF1; this is retained as a negative result.

Next development sequence:

1. vector-native residual/JVP plane and Robertson repeated steps;
2. BDF2/Rosenbrock-W repeated implicit portfolio and Van der Pol/chirped oscillator;
3. W=4 vector short-window residual/JVP and block preconditioners;
4. topology-active 2--4D reduced space;
5. vector current-IVP low-rank preconditioner;
6. executable device-resident CandidateBundle, fused residual and GPU H0 primitive;
7. runtime RHS specialization/cache;
8. topology-guided shared-linearization/block-Krylov experiment;
9. bounded conditional CUDA Graph speculative epoch;
10. mixed-precision speculation with FP64 validation.

## Final rules

Residuals precede losses; ODE is mainline and DAE research-only; topology is core but not a certifier; homotopy is a solver; EESS is reduced-space only; online learning does not require pretrained solver weights; device residency is preferred to CPU/GPU ping-pong; shared candidate/path work should be reused whenever valid; Python stays outside hot loops; GPU/CPU residual parity is mandatory; and the deterministic classical fallback always remains available.
