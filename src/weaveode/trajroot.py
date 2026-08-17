"""Sequential and short-window implicit trajectory root solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from .contracts import FloatArray, Jacobian, Rhs, as_state, positive_step, tolerance_scale, wrms_norm
from .integrators import bdf1_residual
from .nonlinear import NewtonResult, ResidualJvp, newton_solve
from .telemetry import WorkCounter


@dataclass(slots=True)
class StepRootResult:
    y: FloatArray
    accepted: bool
    scaled_residual: float
    message: str
    nonlinear: NewtonResult
    work: WorkCounter


@dataclass(slots=True)
class WindowRootResult:
    y: FloatArray
    accepted: bool
    scaled_residual: float
    message: str
    nonlinear: NewtonResult
    work: WorkCounter


def _combine_work(*counters: WorkCounter) -> WorkCounter:
    total = WorkCounter()
    for counter in counters:
        total += counter
    return total


def solve_bdf1_step(
    rhs: Rhs, t: float, y_prev: npt.ArrayLike, h: float, *, predictor: npt.ArrayLike | None = None,
    jacobian: Jacobian | None = None,
    rhs_jvp: Callable[[float, FloatArray, FloatArray], FloatArray] | None = None,
    atol: float | npt.ArrayLike = 1e-10, rtol: float = 1e-8, root_tol: float = 1e-13,
    certification_tol: float = 1e-9,
) -> StepRootResult:
    h = positive_step(h)
    previous = as_state(y_prev, name="y_prev")
    local_work = WorkCounter()
    if predictor is None:
        derivative = np.asarray(rhs(float(t), previous.copy()), dtype=np.float64)
        local_work.rhs_evaluations += 1
        if derivative.shape != previous.shape or not np.all(np.isfinite(derivative)):
            raise ValueError("RHS returned an invalid predictor derivative")
        initial = previous + h * derivative
    else:
        initial = as_state(predictor, name="predictor")
        if initial.shape != previous.shape:
            raise ValueError("predictor and y_prev must have the same shape")
    def residual(candidate: FloatArray) -> FloatArray:
        local_work.rhs_evaluations += 1
        return bdf1_residual(rhs, t, previous, h, candidate)
    residual_jacobian = None
    residual_jvp: ResidualJvp | None = None
    if jacobian is not None:
        def residual_jacobian(candidate: FloatArray) -> FloatArray:
            jf = np.asarray(jacobian(float(t) + h, candidate.copy()), dtype=np.float64)
            return np.eye(candidate.size, dtype=np.float64) - h * jf
    elif rhs_jvp is not None:
        def residual_jvp(candidate: FloatArray, vector: FloatArray) -> FloatArray:
            return vector - h * np.asarray(rhs_jvp(float(t) + h, candidate.copy(), vector.copy()), dtype=np.float64)
    nonlinear = newton_solve(residual, initial, jacobian=residual_jacobian, jvp=residual_jvp, tol=root_tol)
    final_residual = bdf1_residual(rhs, t, previous, h, nonlinear.x)
    local_work.rhs_evaluations += 1
    scale = tolerance_scale(previous, nonlinear.x, atol, rtol)
    score = wrms_norm(final_residual, scale)
    accepted = bool(nonlinear.converged and np.isfinite(score) and score <= certification_tol)
    message = "accepted" if accepted else f"not certified: {nonlinear.message}, score={score:.3e}"
    return StepRootResult(nonlinear.x.copy(), accepted, score, message, nonlinear, _combine_work(local_work, nonlinear.work))


def solve_implicit_euler_window(
    rhs: Rhs, t0: float, y0: npt.ArrayLike, h: float, steps: int, *, initial: npt.ArrayLike | None = None,
    jacobian: Jacobian | None = None, atol: float | npt.ArrayLike = 1e-10, rtol: float = 1e-8,
    root_tol: float = 1e-12, certification_tol: float = 1e-8,
) -> WindowRootResult:
    h = positive_step(h)
    if steps <= 0: raise ValueError("steps must be positive")
    first = as_state(y0, name="y0")
    dimension = first.size
    local_work = WorkCounter()
    if initial is None:
        guesses: list[FloatArray] = []; state = first.copy()
        for index in range(steps):
            derivative = np.asarray(rhs(float(t0) + index*h, state.copy()), dtype=np.float64)
            local_work.rhs_evaluations += 1
            if derivative.shape != first.shape or not np.all(np.isfinite(derivative)):
                raise ValueError("RHS returned an invalid predictor derivative")
            state = state + h*derivative; guesses.append(state.copy())
        initial_flat = np.concatenate(guesses)
    else:
        initial_array = np.asarray(initial, dtype=np.float64)
        if initial_array.shape == (steps, dimension): initial_flat = np.ascontiguousarray(initial_array.reshape(-1))
        elif initial_array.shape == (steps*dimension,): initial_flat = np.ascontiguousarray(initial_array)
        else: raise ValueError("initial window guess has an invalid shape")
    def residual(flat: FloatArray) -> FloatArray:
        states = flat.reshape(steps, dimension); blocks=[]; previous=first
        for index,current in enumerate(states):
            local_work.rhs_evaluations += 1
            blocks.append(bdf1_residual(rhs,float(t0)+index*h,previous,h,current)); previous=current
        return np.concatenate(blocks)
    residual_jacobian=None
    if jacobian is not None:
        def residual_jacobian(flat: FloatArray) -> FloatArray:
            states=flat.reshape(steps,dimension); matrix=np.zeros((steps*dimension,steps*dimension),dtype=np.float64); eye=np.eye(dimension,dtype=np.float64)
            for index,current in enumerate(states):
                jf=np.asarray(jacobian(float(t0)+(index+1)*h,current.copy()),dtype=np.float64)
                if jf.shape!=(dimension,dimension): raise ValueError("Jacobian returned an invalid shape")
                row=slice(index*dimension,(index+1)*dimension); matrix[row,row]=eye-h*jf
                if index>0: matrix[row,slice((index-1)*dimension,index*dimension)]=-eye
            return matrix
    nonlinear=newton_solve(residual,initial_flat,jacobian=residual_jacobian,tol=root_tol,max_iterations=40)
    states=nonlinear.x.reshape(steps,dimension); full=np.vstack([first,states]); scaled_blocks=[]; previous=first
    for index,current in enumerate(states):
        local_work.rhs_evaluations+=1; block=bdf1_residual(rhs,float(t0)+index*h,previous,h,current); scaled_blocks.append(block/tolerance_scale(previous,current,atol,rtol)); previous=current
    score=float(np.sqrt(np.mean(np.concatenate(scaled_blocks)**2))); accepted=bool(nonlinear.converged and np.isfinite(score) and score<=certification_tol); message="accepted" if accepted else f"not certified: {nonlinear.message}, score={score:.3e}"
    return WindowRootResult(full,accepted,score,message,nonlinear,_combine_work(local_work,nonlinear.work))
