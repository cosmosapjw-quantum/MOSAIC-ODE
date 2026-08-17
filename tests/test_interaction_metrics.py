import math
from weaveode.interaction import pairwise_log_cost_interaction
def test_pairwise_log_cost_interaction_zero_for_multiplicative_independence():assert math.isclose(pairwise_log_cost_interaction(100,80,50,40),0,abs_tol=1e-15)
def test_pairwise_log_cost_interaction_positive_for_superadditive_gain():assert pairwise_log_cost_interaction(100,80,50,30)>0
