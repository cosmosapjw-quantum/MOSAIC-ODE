import math
import numpy as np
from scipy.linalg import expm
from weaveode.integrators import bdf1_residual,exponential_midpoint_step,rosenbrock_euler_step,solve_dopri5
def test_dopri5_decay_accuracy():
 r=solve_dopri5(lambda _t,y:-y,(0.,1.),np.array([1.]),rtol=1e-8,atol=1e-10,initial_step=.1);assert r.success;assert abs(r.y[-1,0]-math.exp(-1))<2e-7
def test_bdf1_linear_exact_discrete_root():
 lam=-13.;yp=np.array([2.,-1.]);h=.03;expected=yp/(1-h*lam);np.testing.assert_allclose(bdf1_residual(lambda _t,y:lam*y,0.,yp,h,expected),0,atol=1e-15)
def test_rosenbrock_euler_is_implicit_euler_for_linear_problem():
 lam=-40.;y=np.array([1.5]);h=.02;got=rosenbrock_euler_step(lambda _t,s:lam*s,lambda _t,_s:np.array([[lam]]),0.,y,h);np.testing.assert_allclose(got,y/(1-h*lam),atol=1e-14)
def test_exponential_midpoint_constant_rotation():
 omega=17.;m=np.array([[0.,-omega],[omega,0.]]);y=np.array([1.,-.25]);h=.07;np.testing.assert_allclose(exponential_midpoint_step(lambda _t:m,0.,y,h),expm(h*m)@y,atol=1e-13)
