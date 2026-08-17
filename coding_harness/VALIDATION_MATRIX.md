# VALIDATION_MATRIX.md

| Requirement | Test/check | Level | Expected | Status | Evidence |
|---|---|---|---|---|---|
| Python import | `python -c 'import weaveode'` | software | pass | PASS | final validation log |
| Native C++ extension | build + import + C API smoke | software | pass | PASS | `tests/test_native.py`, CTest |
| DOPRI5 accuracy | analytic decay test | scientific | within tolerance | PASS | `test_dopri5_decay_accuracy` |
| BDF1 residual authority | linear discrete-root test | scientific | exact to tolerance | PASS | `test_bdf1_linear_exact_discrete_root` |
| Window equivalence | sequential vs W=4 TrajRoot | scientific | agree | PASS | `test_windowed_implicit_euler_matches_sequential_roots` |
| Oscillatory expert | constant rotation test | scientific | phase/state agree | PASS | `test_exponential_midpoint_constant_rotation` |
| Homotopy fold | pseudo-arclength benchmark | numerical | passes fold | PASS | `test_pseudo_arclength_tracks_fold` |
| Function-only path | telemetry audit | numerical | zero J/JVP calls | PASS | `test_function_only_path_uses_no_jacobian_or_jvp` |
| Persistent H0/H1 | two modes + circle tests | scientific | expected topology | PASS | `tests/test_topology_eess.py` |
| EESS mode preservation | two-root population test | scientific | both modes retained | PASS | `test_topology_preserving_selection_keeps_modes` |
| EESS capacity failure | too many components | scientific | explicit failure | PASS | `test_eess_fails_explicitly_when_components_exceed_capacity` |
| Online cold start | zero-output + bounded update | software | pass | PASS | `test_online_adapter_is_zero_at_cold_start_and_bounded_after_update` |
| Fallback learning | classical fallback accepted | scientific | online update retained | PASS | `test_solver_falls_back_when_integrated_candidate_builder_is_empty` |
| Integrated solve | multibranch scalar BDF1 | scientific | principal branch, no false acceptance | PASS | `test_integrated_solver_selects_principal_bdf1_branch` |
| Native/Python parity | fixed candidate batch | numerical | <=1e-12 | PASS | `tests/test_native.py` |
| Fixed-seed reproducibility | benchmark rerun | operational | identical non-timing outputs | PASS | `artifacts/generated/reproducibility.json` |
| Seed sweep | seeds 0--4 | operational | recorded, no false acceptance | PASS | `artifacts/generated/seed_sweep.json` |
| Synergy ladder | B, B+E, B+T, B+E+T+H+M | operational | recorded, no hidden cost | PASS | `artifacts/generated/integrated_benchmark.json` |
| Integrated performance gain | work vs direct backbone | operational | measured, not assumed | CONCERN | integrated work ratio 22.4; no gain |
| CUDA optional source | configure without CUDA | software | CPU build unaffected | PASS | default CMake build; `.cu` source present |
| CUDA execution | compile and device benchmark | performance | execute | NOT_APPLICABLE | no CUDA compiler/device in environment |
| Sanitizers/static checks | ASan/UBSan + warnings + `git diff --check` | software | clean | PASS | final validation log |
| Python line coverage | pytest-cov | software | report | PASS | `artifacts/generated/coverage.json`, ~84% |
| Coding harness | `make harness-check` | operational | pass | PASS | final validation log |

Status: `PASS / CONCERN / FAIL / NOT_RUN / NOT_APPLICABLE`
