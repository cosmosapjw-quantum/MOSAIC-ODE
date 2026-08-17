"""Continuation paths for regular, fold, and function-only root families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt

from .contracts import FloatArray, as_state
from .nonlinear import ResidualJacobian, newton_solve
from .telemetry import WorkCounter

ParameterizedResidual = Callable[[FloatArray, float], FloatArray]
ParameterizedJacobian = Callable[[FloatArray, float], FloatArray]
CurveResidual = Callable[[FloatArray], FloatArray]
ScalarHomotopy = Callable[[float, float], float]


@dataclass(slots=True)
class HomotopyTrace:
    parameters: FloatArray
    states: FloatArray
    converged: bool
    message: str
    derivative_mode: str
    work: WorkCounter


def _sum_work(total: WorkCounter, addition: WorkCounter) -> None:
    total += addition


def continuation_newton(
    residual: ParameterizedResidual,
    start_root: npt.ArrayLike,
    parameters: Iterable[float],
    *,
    jacobian: ParameterizedJacobian | None = None,
    tol: float = 1e-11,
    max_iterations: int = 30,
) -> HomotopyTrace:
    values = np.asarray(list(parameters), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("parameters must be a non-empty finite sequence")
    state = as_state(start_root, name="start_root")
    work = WorkCounter()
    states = [state.copy()]
    initial = np.asarray(residual(state.copy(), float(values[0])), dtype=np.float64)
    work.residual_evaluations += 1
    if initial.shape != state.shape or not np.all(np.isfinite(initial)):
        raise ValueError("residual returned an invalid initial value")
    if np.linalg.norm(initial, ord=np.inf) > tol:
        return HomotopyTrace(values[:1], np.vstack(states), False, "start_root does not satisfy first parameter", "J" if jacobian else "FD_J", work)
    for index in range(1, values.size):
        parameter = float(values[index])
        def local_residual(candidate: FloatArray) -> FloatArray:
            return np.asarray(residual(candidate, parameter), dtype=np.float64)
        local_jacobian: ResidualJacobian | None = None
        if jacobian is not None:
            def local_jacobian(candidate: FloatArray) -> FloatArray:
                return np.asarray(jacobian(candidate, parameter), dtype=np.float64)
        result = newton_solve(local_residual, state, jacobian=local_jacobian, tol=tol, max_iterations=max_iterations)
        _sum_work(work, result.work)
        if not result.converged:
            return HomotopyTrace(values[:index], np.vstack(states), False, f"continuation failed at parameter {parameter}: {result.message}", "J" if jacobian else "FD_J", work)
        state = result.x.copy()
        states.append(state)
    return HomotopyTrace(values, np.vstack(states), True, "converged", "J" if jacobian else "FD_J", work)


def pseudo_arclength_curve(
    residual: CurveResidual,
    start_points: npt.ArrayLike,
    *,
    step_size: float,
    steps: int,
    tol: float = 1e-10,
    max_iterations: int = 30,
) -> HomotopyTrace:
    points = np.asarray(start_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 2:
        raise ValueError("start_points must have shape (2, n) with n >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError("start_points must be finite")
    if not np.isfinite(step_size) or step_size <= 0.0 or steps < 0:
        raise ValueError("step_size must be positive and steps nonnegative")
    work = WorkCounter()
    path = [points[0].copy(), points[1].copy()]
    parameter = [0.0, float(np.linalg.norm(points[1] - points[0]))]
    for point in path:
        value = np.asarray(residual(point.copy()), dtype=np.float64)
        work.residual_evaluations += 1
        if value.shape != (points.shape[1] - 1,) or not np.all(np.isfinite(value)):
            raise ValueError("curve residual must return n-1 finite values")
        if np.linalg.norm(value, ord=np.inf) > 100.0 * tol:
            return HomotopyTrace(np.asarray(parameter), np.vstack(path), False, "starting point is off the zero set", "FD_J", work)
    for _ in range(steps):
        tangent = path[-1] - path[-2]
        norm = float(np.linalg.norm(tangent))
        if norm == 0.0:
            return HomotopyTrace(np.asarray(parameter), np.vstack(path), False, "zero secant tangent", "FD_J", work)
        tangent /= norm
        predictor = path[-1] + step_size * tangent
        def augmented(candidate: FloatArray) -> FloatArray:
            base = np.asarray(residual(candidate.copy()), dtype=np.float64)
            plane = np.array([np.dot(candidate - predictor, tangent)], dtype=np.float64)
            return np.concatenate([base, plane])
        result = newton_solve(augmented, predictor, tol=tol, max_iterations=max_iterations)
        _sum_work(work, result.work)
        if not result.converged:
            return HomotopyTrace(np.asarray(parameter), np.vstack(path), False, f"pseudo-arclength corrector failed: {result.message}", "FD_J", work)
        corrected = result.x.copy()
        if np.dot(corrected - path[-1], tangent) <= 0.0:
            return HomotopyTrace(np.asarray(parameter), np.vstack(path), False, "corrector reversed path orientation", "FD_J", work)
        path.append(corrected)
        parameter.append(parameter[-1] + float(np.linalg.norm(path[-1] - path[-2])))
    return HomotopyTrace(np.asarray(parameter), np.vstack(path), True, "converged", "FD_J", work)


def _evaluate_scalar(function: ScalarHomotopy, x: float, parameter: float, work: WorkCounter) -> float:
    value = float(function(float(x), float(parameter)))
    work.residual_evaluations += 1
    if not np.isfinite(value):
        raise FloatingPointError("function-only homotopy returned NaN or Inf")
    return value


def function_only_scalar_path(
    function: ScalarHomotopy,
    *,
    start_x: float,
    lambdas: Iterable[float],
    initial_radius: float,
    tol: float = 1e-10,
    grid_points: int = 65,
    max_expansions: int = 8,
) -> HomotopyTrace:
    parameters = np.asarray(list(lambdas), dtype=np.float64)
    if parameters.ndim != 1 or parameters.size == 0 or not np.all(np.isfinite(parameters)):
        raise ValueError("lambdas must be a non-empty finite sequence")
    if initial_radius <= 0.0 or not np.isfinite(initial_radius):
        raise ValueError("initial_radius must be positive and finite")
    if grid_points < 3 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer at least 3")
    work = WorkCounter()
    current = float(start_x)
    initial_value = _evaluate_scalar(function, current, float(parameters[0]), work)
    if abs(initial_value) > tol:
        return HomotopyTrace(parameters[:1], np.array([[current]], dtype=np.float64), False, "start_x does not satisfy first lambda", "FUNCTION_ONLY", work)
    states = [current]
    for index in range(1, parameters.size):
        parameter = float(parameters[index])
        bracket: tuple[float, float, float, float] | None = None
        exact: float | None = None
        radius = float(initial_radius)
        for _ in range(max_expansions + 1):
            grid = np.linspace(current - radius, current + radius, grid_points)
            values = np.array([_evaluate_scalar(function, float(x), parameter, work) for x in grid], dtype=np.float64)
            exact_indices = np.flatnonzero(np.abs(values) <= tol)
            if exact_indices.size:
                exact = float(grid[int(exact_indices[np.argmin(np.abs(grid[exact_indices] - current))])])
                break
            candidates: list[tuple[float, float, float, float]] = []
            for left in range(grid_points - 1):
                if np.signbit(values[left]) != np.signbit(values[left + 1]):
                    candidates.append((float(grid[left]), float(grid[left + 1]), float(values[left]), float(values[left + 1])))
            if candidates:
                bracket = min(candidates, key=lambda item: abs(0.5 * (item[0] + item[1]) - current))
                break
            radius *= 2.0
        if exact is not None:
            current = exact
            states.append(current)
            continue
        if bracket is None:
            return HomotopyTrace(parameters[:index], np.asarray(states, dtype=np.float64)[:, None], False, f"no sign-changing bracket at lambda={parameter}", "FUNCTION_ONLY", work)
        left, right, f_left, f_right = bracket
        root = 0.5 * (left + right)
        for _ in range(100):
            root = 0.5 * (left + right)
            f_root = _evaluate_scalar(function, root, parameter, work)
            if abs(f_root) <= tol or 0.5 * (right - left) <= tol:
                break
            if np.signbit(f_left) != np.signbit(f_root):
                right = root
                f_right = f_root
            else:
                left = root
                f_left = f_root
        current = root
        states.append(current)
    return HomotopyTrace(parameters, np.asarray(states, dtype=np.float64)[:, None], True, "converged", "FUNCTION_ONLY", work)
