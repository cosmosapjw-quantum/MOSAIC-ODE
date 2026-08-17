from __future__ import annotations
import math
import numpy as np
from weaveode.online import OnlineLowRankAdapter
from weaveode.problems import PolynomialScalarProblem
from weaveode.solver import WeaveBDF1Solver

def test_online_adapter_is_zero_at_cold_start_and_bounded_after_update():
    adapter=OnlineLowRankAdapter(state_dimension=1,feature_dimension=6,rank=3,learning_rate=.08,max_relative_correction=.2,seed=7);features=np.array([.5,.1,.25,.525,1.,2.]);scale=np.array([2.]);np.testing.assert_array_equal(adapter.propose(features,scale),np.zeros(1))
    for _ in range(40): assert adapter.update(features,np.array([.3]),scale).accepted
    assert 0<adapter.propose(features,scale)[0]<=.4

def test_integrated_solver_selects_principal_bdf1_branch():
    problem=PolynomialScalarProblem(np.array([0.,0.,1.]));solver=WeaveBDF1Solver(problem.rhs,polynomial_coefficients=problem.coefficients,jacobian=problem.jacobian,step_size=.1,search_radius=12.,grid_candidates=129,eess_points=16,component_radius=.5,seed=3);result=solver.step(0.,np.array([.25]));expected=(1-math.sqrt(1-4*.1*.25))/(2*.1);assert result.accepted;assert abs(result.y[0]-expected)<1e-10;assert result.trace.component_count>=2

def test_integrated_multistep_solve_keeps_initial_condition_and_updates_online():
    p=PolynomialScalarProblem(np.array([0.,-2.]));s=WeaveBDF1Solver(p.rhs,polynomial_coefficients=p.coefficients,jacobian=p.jacobian,step_size=.05,grid_candidates=33,search_radius=.5,seed=11);trace=s.solve((0.,.25),np.array([1.]));assert trace.success;assert trace.y[0,0]==1.;assert trace.work.online_updates>0
