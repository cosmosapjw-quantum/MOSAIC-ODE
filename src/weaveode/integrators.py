"""Minimal classical method bank used by the integrated vertical slice."""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from .contracts import FloatArray,IntegrationResult,Jacobian,MatrixGenerator,Rhs,as_state,positive_step,tolerance_scale,wrms_norm
from .telemetry import WorkCounter
def _rhs_value(rhs,t,y,work):
    value=np.asarray(rhs(float(t),y.copy()),dtype=np.float64); work.rhs_evaluations+=1
    if value.shape!=y.shape or not np.all(np.isfinite(value)): raise ValueError("invalid RHS")
    return np.ascontiguousarray(value)
def _dopri5_trial(rhs,t,y,h,work):
    k1=_rhs_value(rhs,t,y,work); k2=_rhs_value(rhs,t+h/5,y+h*k1/5,work); k3=_rhs_value(rhs,t+3*h/10,y+h*(3*k1/40+9*k2/40),work); k4=_rhs_value(rhs,t+4*h/5,y+h*(44*k1/45-56*k2/15+32*k3/9),work); k5=_rhs_value(rhs,t+8*h/9,y+h*(19372*k1/6561-25360*k2/2187+64448*k3/6561-212*k4/729),work); k6=_rhs_value(rhs,t+h,y+h*(9017*k1/3168-355*k2/33+46732*k3/5247+49*k4/176-5103*k5/18656),work); y5=y+h*(35*k1/384+500*k3/1113+125*k4/192-2187*k5/6784+11*k6/84); k7=_rhs_value(rhs,t+h,y5,work); y4=y+h*(5179*k1/57600+7571*k3/16695+393*k4/640-92097*k5/339200+187*k6/2100+k7/40); return y5,y5-y4
def solve_dopri5(rhs:Rhs,t_span:tuple[float,float],y0:npt.ArrayLike,*,rtol:float=1e-6,atol:float|npt.ArrayLike=1e-9,initial_step:float|None=None,max_steps:int=100000,min_step:float=1e-15,safety:float=.9)->IntegrationResult:
    t0,t_end=map(float,t_span); y=as_state(y0,name="y0"); work=WorkCounter(); h=positive_step(initial_step if initial_step is not None else min(1e-2,t_end-t0)); times=[t0]; states=[y.copy()]; t=t0; attempts=0
    while t<t_end:
        if attempts>=max_steps: return IntegrationResult(np.asarray(times),np.vstack(states),False,"maximum step count exceeded",work)
        attempts+=1; h=min(h,t_end-t)
        if h<min_step or t+h==t: return IntegrationResult(np.asarray(times),np.vstack(states),False,"step size underflow",work)
        trial,error=_dopri5_trial(rhs,t,y,h,work); error_norm=wrms_norm(error,tolerance_scale(y,trial,atol,rtol))
        if error_norm<=1.0: t+=h; y=trial; times.append(t); states.append(y.copy()); work.accepted_steps+=1; h*=float(np.clip(5 if error_norm==0 else safety*error_norm**(-.2),.2,5))
        else: work.rejected_steps+=1; h*=float(np.clip(safety*error_norm**(-.2),.1,.5))
    return IntegrationResult(np.asarray(times),np.vstack(states),True,"success",work)
def bdf1_residual(rhs:Rhs,t:float,y_prev:npt.ArrayLike,h:float,candidate:npt.ArrayLike)->FloatArray:
    h=positive_step(h); previous=as_state(y_prev,name="y_prev"); current=as_state(candidate,name="candidate"); value=np.asarray(rhs(float(t)+h,current.copy()),dtype=np.float64); return current-previous-h*value
def rosenbrock_euler_step(rhs:Rhs,jacobian:Jacobian,t:float,y:npt.ArrayLike,h:float)->FloatArray:
    h=positive_step(h); state=as_state(y); f=np.asarray(rhs(float(t),state.copy()),dtype=np.float64); jac=np.asarray(jacobian(float(t),state.copy()),dtype=np.float64); return state+h*np.linalg.solve(np.eye(state.size)-h*jac,f)
def exponential_midpoint_step(generator:MatrixGenerator,t:float,y:npt.ArrayLike,h:float)->FloatArray:
    h=positive_step(h); state=as_state(y); matrix=np.asarray(generator(float(t)+.5*h),dtype=np.float64); return np.asarray(expm(h*matrix)@state,dtype=np.float64)
