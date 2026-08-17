# MOSAIC-ODE project instructions

## Research identity

MOSAIC-ODE is an integrated solver product, not a collection of unrelated papers. Mature methods should be reused where possible. Experimental topology/ML/GPU ideas are added because they may improve the integrated solve, not because each must be independently novel.

## Current next-development targets

1. vector repeated implicit steps;
2. Robertson and large-mu Van der Pol;
3. topology-active reduced candidate spaces;
4. current-IVP low-rank preconditioning;
5. executable CUDA CandidateBundle and fused scoring;
6. topology-guided shared-linearization/block-Krylov experiments;
7. runtime RHS specialization;
8. CUDA Graph speculative epoch after kernel contracts stabilize.

## Approval-sensitive changes

Do not silently change:

- mainline ODE scope;
- DAE research-only status;
- original residual/error acceptance semantics;
- FP64 validation requirement for accepted GPU proposals;
- topology/homotopy role separation;
- implementation-language constraint.
