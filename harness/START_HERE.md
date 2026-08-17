# Start here — MOSAIC-ODE integrated harness

1. Read `../AGENTS.md`.
2. Read `SCIENTIFIC_CONTRACT.md` and `VALIDATION_MATRIX.md`.
3. Read `../docs/design/MOSAIC_ODE_INTEGRATED_PREPRODUCT_DESIGN.md`.
4. Select the narrowest repository skill under `../.agents/skills/`.
5. Reproduce the baseline before editing.
6. Preserve interaction hypotheses: do not delete a weak isolated module before the selected combination tests are run.
7. After mutation, run `../scripts/verify_preproduct.sh` and an independent diff review.
