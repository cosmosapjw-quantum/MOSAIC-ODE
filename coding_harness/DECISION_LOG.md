# DECISION_LOG.md

## D-001

DATE:

DECISION: `promote / hold / rework / revert`

OBJECT:

EVIDENCE:

RATIONALE:

ALTERNATIVES:

RISKS:

NEXT_ACTION:

## 2026-08-17 — Roundoff-aware test tolerance for extremely small WRMS scales

The linear window-equivalence test uses `atol=1e-13`, `rtol=1e-11`. A backward-Euler root with an absolute floating residual of approximately `1e-16` therefore has a scaled WRMS score of order `1e-6`; demanding `1e-9` would require an absolute cancellation below binary64 roundoff. The test explicitly configures `certification_tol=1e-5` while separately requiring the discrete state to agree within `2e-12`. Production acceptance remains governed by the user-configured original-residual tolerance; no hidden bypass was added.
