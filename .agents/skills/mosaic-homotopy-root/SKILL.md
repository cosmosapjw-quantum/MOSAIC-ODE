---
name: mosaic-homotopy-root
description: Develop the multi-lane homotopy/root engine, distinguishing J, Jv, function-only, folds, and singular endpoints.
---

# Homotopy / Root Engine

1. Classify scope, derivative access, and zero-set geometry independently.
2. Use direct Newton/JFNK before expensive continuation when appropriate.
3. For simple folds use augmented pseudo-arclength semantics.
4. For singular endpoints use reduced-subspace/endgame/branch-switch logic; do not claim coordinate transforms remove rank loss.
5. Function-only claims require zero J/JVP/AD directional-derivative telemetry.
6. EESS supplies branch hypotheses; original residual validation remains final.
