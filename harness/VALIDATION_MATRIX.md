# MOSAIC-ODE Validation Matrix

| Layer | Required checks | Bootstrap status target |
|---|---|---|
| C ABI | build, smoke, invalid input | pass |
| Python facade | import, legacy identity | pass |
| Classical ODE | DOPRI5/BDF1/Rosenbrock analytic checks | pass |
| Vector native | vector BDF1 residual scoring parity | pass |
| Window solve | sequential discrete-root equivalence | pass where implemented |
| Homotopy | regular path, simple fold, function-only telemetry | pass |
| Topology | H0 components, H1 circle regression | pass |
| EESS | component preservation, bounded state transport | pass |
| Online learning | cold start, bounded update, rollback | pass |
| Integrated vector slice | certified vector BDF1 root | pass |
| CUDA source | contract/source presence and CPU parity design | source-only unless CUDA available |
| CUDA execution | build + GPU parity + timing | blocked when CUDA unavailable |
| GPU performance | single-IVP crossover including overhead | not claimed |
| DAE | no mainline claim | research-only |
| Packaging | harness validator, archive integrity, git bundle verify | pass |
