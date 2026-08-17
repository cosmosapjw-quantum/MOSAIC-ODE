# SCIENTIFIC_CONTRACT.md

## Scientific objective

Build and test the first integrated C/C++/Python vertical slice of **WeaveODE**: a residual-first ODE polyalgorithm in which a classical predictor/corrector, short-window trajectory root solve, persistent EESS population, topology-aware mode preservation, homotopy root paths, a cold-start online adapter, and an optional native/CUDA candidate evaluator participate in one solve while the original discretized residual remains the acceptance authority.

## Governing definitions

Mainline problem:

\[
\dot y=f(t,y),\qquad y(t_0)=y_0.
\]

The first implicit host is backward Euler/BDF1:

\[
R_n(Y)=Y-y_n-h f(t_{n+1},Y)=0.
\]

A length-\(W\) TrajRoot window uses

\[
R_{n:n+W}(z)=\bigl(R_n(y_{n+1}),\ldots,R_{n+W-1}(y_{n+W})\bigr)^T=0.
\]

EESS candidates are finite root/trajectory hypotheses scored by a dimensionless residual pseudo-observation model. Topology observes the candidate sublevel-set geometry; homotopy solves zero paths; topology does not certify roots.

## Conventions

- State vectors are real-valued NumPy/PyTorch arrays with final axis equal to state dimension.
- Time increases forward and every step satisfies `h > 0`.
- Residual scaling is componentwise `atol + rtol * max(abs(reference), abs(candidate))` unless a problem-specific positive scale is supplied.
- Final acceptance uses FP64 original-space residuals.
- The online model is cold-started and zero-output at the beginning of every IVP.
- Fixed-seed stochastic runs use NumPy and PyTorch seeds recorded in telemetry.

## Valid regime

- Mainline: explicit first-order ODEs, including stiff and oscillatory test problems.
- Integrated V0 implicit host: BDF1 and a fixed short window, with small or medium state dimension.
- Derivative-free homotopy: scalar or reduced dimension at most 2.
- Persistent homology implementation: small candidate clouds (default at most 96 points).
- DAE support is out of this coding slice.

## Required invariants

- Exact initial condition: `y(t0) == y0`.
- No candidate is accepted unless the original BDF1 residual satisfies the configured scaled tolerance.
- `FUNCTION_ONLY` homotopy telemetry records zero Jacobian/JVP calls.
- Topology may allocate candidates/paths but cannot mark a step accepted.
- The native and Python candidate scores agree within FP64 tolerance.
- Online adaptation is included in work accounting and can be disabled without breaking the classical solve.

## Known limits

| Limit | Expected result | Tolerance | Reference/test |
|---|---|---:|---|
| `y' = -y`, DOPRI5 | fifth-order solution with embedded control | global error < 2e-7 at `rtol=1e-8` | `test_dopri5_decay_accuracy` |
| `y' = λy`, BDF1 | exact discrete recurrence `y_{n+1}=y_n/(1-hλ)` | 1e-12 | `test_bdf1_linear_exact_discrete_root` |
| Constant skew matrix | exponential midpoint equals matrix exponential | 1e-12 | `test_exponential_midpoint_constant_rotation` |
| Fold `x^2-μ=0` | pseudo-arclength passes the turning point | residual < 1e-8 | `test_pseudo_arclength_tracks_fold` |
| Circle cloud | one persistent H1 class | persistence > threshold | `test_vr_persistence_detects_circle_loop` |
| Two separated modes | topology-aware EESS retains both components | both representatives retained | `test_topology_preserving_selection_keeps_modes` |

## Numerical requirements

- Default dtype: float64.
- Root certification tolerance: scaled residual norm <= 1e-9 in tests unless stated otherwise.
- No NaN/Inf, silent clipping, or unreported fallback.
- CPU native extension and pure-Python fallback must be numerically consistent.
- CUDA source is optional; no GPU performance claim is permitted unless compiled and run on a CUDA device.

## Failure semantics

NaN/Inf, non-convergence, empty candidate set, missing topology representative, failed native build, and failed residual certification are explicit failures. A classical fallback may recover a step, but telemetry must record it.

## Change control

Changing the residual definition, branch admissibility rule, error scaling, benchmark tolerances, or accepted-step semantics requires explicit review in the decision log.
