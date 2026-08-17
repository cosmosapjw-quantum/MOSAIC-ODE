"""Minimal classical method bank used by the integrated vertical slice."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from .contracts import FloatArray, IntegrationResult, Jacobian, MatrixGenerator, Rhs, as_state, positive_step, tolerance_scale, wrms_norm
from .telemetry import WorkCounter


def _rhs_value(rhs: Rhs, t: float, y: FloatArray, work: WorkCounter) -> FloatArray:
    value = np.asarray(rhs(float(t), y.copy()), dtype=np.float64)
    work.rhs_evaluations += 1
    if value.shape != y.shape:
        raise ValueError(f"RHS returned shape {value.shape}, expected {y.shape}")
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("RHS returned NaN or Inf")
    return np.ascontiguousarray(value)


def _dopri5_trial(rhs: Rhs, t: float, y: FloatArray, h: float, work: WorkCounter) -> tuple[FloatArray, FloatArray]:
    k1 = _rhs_value(rhs, t, y, work)
    k2 = _rhs_value(rhs, t + h*(1/5), y + h*((1/5)*k1), work)
    k3 = _rhs_value(rhs, t + h*(3/10), y + h*((3/40)*k1 + (9/40)*k2), work)
    k4 = _rhs_value(rhs, t + h*(4/5), y + h*((44/45)*k1 - (56/15)*k2 + (32/9)*k3), work)
    k5 = _rhs_value(rhs, t + h*(8/9), y + h*((19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4), work)
    k6 = _rhs_value(rhs, t + h, y + h*((9017/3168)*k1 - (355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5), work)
    y5 = y + h*((35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6)
    k7 = _rhs_value(rhs, t + h, y5, work)
    y4 = y + h*((5179/57600)*k1 + (7571/16695)*k3 + (393/640)*k4 - (92097/339200)*k5 + (187/2100)*k6 + (1/40)*k7)
    return y5, y5-y4


def solve_dopri5(rhs: Rhs, t_span: tuple[float,float], y0: npt.ArrayLike, *, rtol: float=1e-6, atol: float|npt.ArrayLike=1e-9, initial_step: float|None=None, max_steps:int=100000, min_step:float=1e-15, safety:float=.9) -> IntegrationResult:
    t0,t_end=map(float,t_span)
    if not np.isfinite(t0) or not np.isfinite(t_end) or not t_end>t0:
        raise ValueError("t_span must be finite and increasing")
    if max_steps<=0:
        raise ValueError("max_steps must be positive")
    y=as_state(y0,name="y0"); work=WorkCounter(); interval=t_end-t0; h=positive_step(initial_step if initial_step is not None else min(1e-2,interval)); h=min(h,interval); times=[t0]; states=[y.copy()]; t=t0; attempts=0
    while t<t_end:
        if attempts>=max_steps: return IntegrationResult(np.asarray(times),np.vstack(states),False,"maximum step count exceeded",work)
        attempts+=1; h=min(h,t_end-t)
        if h<min_step or t+h==t: return IntegrationResult(np.asarray(times),np.vstack(states),False,"step size underflow",work)
        trial,error=_dopri5_trial(rhs,t,y,h,work); scale=tolerance_scale(y,trial,atol,rtol); error_norm=wrms_norm(error,scale)
        if not np.isfinite(error_norm): return IntegrationResult(np.asarray(times),np.vstack(states),False,"non-finite error estimate",work)
        if error_norm<=1.0:
            t=t+h; y=trial; times.append(t); states.append(y.copy()); work.accepted_steps+=1; factor=5.0 if error_norm==0 else safety*error_norm**(-.2); h*=float(np.clip(factor,.2,5.0))
        else:
            work.rejected_steps+=1; factor=safety*error_norm**(-.2); h*=float(np.clip(factor,.1,.5))
    return IntegrationResult(np.asarray(times),np.vstack(states),True,"success",work)


def bdf1_residual(rhs: Rhs, t: float, y_prev: npt.ArrayLike, h: float, candidate: npt.ArrayLike) -> FloatArray:
    h=positive_step(h); previous=as_state(y_prev,name="y_prev"); current=as_state(candidate,name="candidate")
    if current.shape!=previous.shape: raise ValueError("candidate and y_prev must have the same shape")
    value=np.asarray(rhs(float(t)+h,current.copy()),dtype=np.float64)
    if value.shape!=current.shape or not np.all(np.isfinite(value)): raise ValueError("RHS returned an invalid value")
    return current-previous-h*value


def rosenbrock_euler_step(rhs: Rhs, jacobian: Jacobian, t: float, y: npt.ArrayLike, h: float) -> FloatArray:
    h=positive_step(h); state=as_state(y); f=np.asarray(rhs(float(t),state.copy()),dtype=np.float64); jac=np.asarray(jacobian(float(t),state.copy()),dtype=np.float64)
    if f.shape!=state.shape or jac.shape!=(state.size,state.size): raise ValueError("RHS or Jacobian returned invalid shape")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(jac)): raise FloatingPointError("RHS or Jacobian returned NaN/Inf")
    try: increment=np.linalg.solve(np.eye(state.size,dtype=np.float64)-h*jac,f)
    except np.linalg.LinAlgError as exc: raise RuntimeError("Rosenbrock stage matrix is singular") from exc
    return state+h*increment


def exponential_midpoint_step(generator: MatrixGenerator, t: float, y: npt.ArrayLike, h: float) -> FloatArray:
    h=positive_step(h); state=as_state(y); matrix=np.asarray(generator(float(t)+.5*h),dtype=np.float64)
    if matrix.shape!=(state.size,state.size): raise ValueError("matrix generator returned an invalid shape")
    if not np.all(np.isfinite(matrix)): raise FloatingPointError("matrix generator returned NaN/Inf")
    return np.asarray(expm(h*matrix)@state,dtype=np.float64)
