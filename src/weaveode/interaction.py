"""Descriptive interaction-gain metrics, separate from likelihood and acceptance."""
from __future__ import annotations
import math
def _positive_cost(name,value):
    value=float(value)
    if not math.isfinite(value) or value<=0: raise ValueError(f"{name} must be positive and finite")
    return value
def pairwise_log_cost_interaction(baseline_cost:float,cost_a:float,cost_b:float,cost_ab:float)->float:
    c0=_positive_cost("baseline_cost",baseline_cost); ca=_positive_cost("cost_a",cost_a); cb=_positive_cost("cost_b",cost_b); cab=_positive_cost("cost_ab",cost_ab)
    return -math.log(cab)+math.log(ca)+math.log(cb)-math.log(c0)
