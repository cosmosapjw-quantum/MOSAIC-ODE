# Current-IVP Online Learning Contract

MOSAIC-ODE must remain functional with no pretrained solver weights.

A cold-start online learner may adapt only compact state such as:

- low-rank preconditioner/correction factors;
- local chart parameters;
- controller hidden state;
- scalar damping/frequency/time-gauge heads;
- homotopy deformation parameters.

Updates are transactional and rollback on non-finite or worsening local objectives. Accepted classical fallback corrections remain valid training signals. Same-IVP adaptation cost is part of performance accounting.
