# CUDA Runtime Architecture

The GPU is a single-IVP internal co-solver, not only a batch accelerator.

## Target device-resident state

- CandidateBundle state/residual/score arrays;
- EESS population and ancestry;
- topology component/path metadata;
- Krylov/multi-RHS workspace;
- compact online-model state.

## Interaction target

\[
\text{topology clustering}
\rightarrow
\text{shared-linearization groups}
\rightarrow
\text{block/multi-RHS Krylov}
\rightarrow
\text{CUDA Graph execution}.
\]

The design seeks to reduce duplicated work, not merely hide an unfavorable work count behind GPU throughput.

## Precision policy

- FP32/TF32: proposal, coarse, online-adapter, or preconditioner work.
- FP64: final original-residual validation and branch-sensitive checks.
- CPU: independent audit/fallback, especially on topology/branch changes or unsupported callbacks.
