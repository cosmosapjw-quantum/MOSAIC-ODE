"""Interaction-gain metrics for integrated solver experiments.

The metrics in this module are descriptive unless uncertainty across problem
instances or seeds is reported. They are deliberately kept separate from
Bayesian EESS likelihoods and from step acceptance.
"""

from __future__ import annotations

import math


def _positive_cost(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def pairwise_log_cost_interaction(
    baseline_cost: float,
    cost_a: float,
    cost_b: float,
    cost_ab: float,
) -> float:
    """Return the factorial-style log-cost interaction.

    Positive values mean the combined configuration costs less than expected
    from multiplying the two isolated cost ratios relative to the baseline.
    """

    c0 = _positive_cost("baseline_cost", baseline_cost)
    ca = _positive_cost("cost_a", cost_a)
    cb = _positive_cost("cost_b", cost_b)
    cab = _positive_cost("cost_ab", cost_ab)
    return -math.log(cab) + math.log(ca) + math.log(cb) - math.log(c0)
