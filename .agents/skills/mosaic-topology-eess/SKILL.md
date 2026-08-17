---
name: mosaic-topology-eess
description: Develop topology-aware EESS behavior while preserving the observer/solver/certifier role separation.
---

# Topology + EESS

1. Operate on a reduced branch/soft space, never unconstrained full-trajectory samples.
2. Maintain cheap component/branch state across steps.
3. Use persistent H0/H1/Reeb information only when it changes allocation or branch handling.
4. Topology may schedule EESS points and homotopy paths but may not accept a numerical step.
5. Compare EESS against Sobol/multistart/deflation on hard cases.
6. Preserve mode representatives when capacity permits; fail explicitly if required topology cannot be represented.
