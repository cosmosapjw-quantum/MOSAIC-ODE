# MOSAIC-ODE Scientific Contract

## Primary problem

Mainline systems are ODE IVPs

\[
\dot y=f(t,y;\theta),\qquad y(t_0)=y_0.
\]

## Fixed semantics

1. A numerical candidate is not accepted because a neural score, topology score, EESS score, or optimizer loss is small.
2. Acceptance is based on the original discrete/original-equation residual, independent discretization/local-error information, hard constraints, and branch continuity.
3. Jacobian-explicit, matrix-free JVP, and strict function-only derivative access are distinct and telemetry-audited.
4. Simple-fold continuation and singular-endpoint handling are distinct.
5. Persistent homology does not imply homotopy equivalence.
6. Bayesian EESS semantics require a declared residual measure/noise model; otherwise use generalized-Bayes/Gibbs terminology.
7. GPU/ML speed claims include launch, transfer, JIT, adaptation, synchronization, and fallback costs.
8. DAE support is research-only in this bootstrap.

## Interaction objective

A component is not rejected solely because its isolated gain is small. Selected pairwise/higher-order combinations are evaluated before removal, especially:

- topology + EESS;
- topology + homotopy;
- topology + path grouping + block Krylov;
- online learning + repeated implicit steps;
- GPU + EESS + homotopy;
- windowed solve + online preconditioning.
