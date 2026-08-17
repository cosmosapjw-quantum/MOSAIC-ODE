import numpy as np
from weaveode.homotopy import continuation_newton,function_only_scalar_path,pseudo_arclength_curve
def test_continuation_newton_tracks_smooth_root_family():
 l=np.linspace(0,1,11);tr=continuation_newton(lambda x,lam:np.array([x[0]**3+x[0]-lam]),np.array([0.]),l,jacobian=lambda x,_:np.array([[3*x[0]**2+1.]]));assert tr.converged;assert abs(tr.states[-1,0]**3+tr.states[-1,0]-1)<1e-11
def test_pseudo_arclength_tracks_fold():
 tr=pseudo_arclength_curve(lambda q:np.array([q[0]**2-q[1]]),np.array([[1.,1.],[.8,.64]]),step_size=.08,steps=30,tol=1e-11);assert tr.converged;assert np.min(tr.states[:,0])<-.25
def test_function_only_path_uses_no_jacobian_or_jvp():
 tr=function_only_scalar_path(lambda x,lam:x**3+x-lam,start_x=0,lambdas=np.linspace(0,1,9),initial_radius=.3,tol=1e-12);assert tr.converged;assert tr.derivative_mode=='FUNCTION_ONLY';assert tr.work.jacobian_evaluations==0;assert tr.work.jvp_evaluations==0
