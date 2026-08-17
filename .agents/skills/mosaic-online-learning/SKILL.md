---
name: mosaic-online-learning
description: Develop current-IVP online adaptation that participates in solving without requiring pretrained solver weights.
---

# Online Learning

1. Cold start must remain valid with zero/neutral learned state.
2. Train only compact bounded state such as low-rank adapters, local chart heads, or controller state.
3. Learn from accepted classical/nonlinear corrections, residual reduction, path survival, and topology consistency.
4. Transactionally rollback non-finite or worsening updates.
5. Include adaptation cost in same-IVP performance accounting.
6. Learned output is proposal/preconditioner/path shaping only until original residual/error validation passes.
