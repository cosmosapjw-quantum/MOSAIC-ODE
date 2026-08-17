import numpy as np
from weaveode.nonlinear import newton_solve
from weaveode.trajroot import solve_bdf1_step,solve_implicit_euler_window
def test_newton_solve_linear_system():
 m=np.array([[3.,1.],[1.,2.]]);target=np.array([1.,-4.]);r=newton_solve(lambda x:m@x-target,np.zeros(2),jacobian=lambda _x:m,tol=1e-13);assert r.converged;np.testing.assert_allclose(r.x,np.linalg.solve(m,target),atol=1e-14)
def test_newton_solve_matrix_free_jvp():
 r=newton_solve(lambda x:np.array([x[0]**3-2]),np.array([1.]),jvp=lambda x,v:np.array([3*x[0]**2*v[0]]),tol=1e-12);assert r.converged;assert r.work.jvp_evaluations>0;assert r.work.jacobian_evaluations==0
def test_bdf1_step_selects_exact_linear_discrete_root():
 lam=-25.;y0=np.array([1.25]);h=.04;r=solve_bdf1_step(lambda _t,y:lam*y,0.,y0,h,jacobian=lambda _t,_y:np.array([[lam]]),atol=1e-12,rtol=1e-10);assert r.accepted;np.testing.assert_allclose(r.y,y0/(1-h*lam),atol=1e-13)
def test_windowed_implicit_euler_matches_sequential_roots():
 lam=-7.;y0=np.array([2.,-.5]);h=.03;steps=4;rhs=lambda _t,y:lam*y;jac=lambda _t,_y:lam*np.eye(2);seq=[y0];cur=y0;t=0.
 for _ in range(steps):
  step=solve_bdf1_step(rhs,t,cur,h,jacobian=jac,atol=1e-13,rtol=1e-11,certification_tol=1e-5);assert step.accepted;cur=step.y;seq.append(cur);t+=h
 win=solve_implicit_euler_window(rhs,0.,y0,h,steps,jacobian=jac,atol=1e-13,rtol=1e-11,certification_tol=1e-5);assert win.accepted;np.testing.assert_allclose(win.y,np.vstack(seq),atol=2e-12)
