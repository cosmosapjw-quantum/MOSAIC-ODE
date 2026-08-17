"""Small nonlinear engines with explicit derivative-access telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator, gmres

from .contracts import FloatArray, as_state
from .telemetry import WorkCounter

Residual = Callable[[FloatArray], FloatArray]
ResidualJacobian = Callable[[FloatArray], FloatArray]
ResidualJvp = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(slots=True)
class NewtonResult:
    x: FloatArray
    converged: bool
    residual_norm: float
    iterations: int
    message: str
    work: WorkCounter


def _finite_difference_jacobian(
    evaluate: Callable[[FloatArray], FloatArray],
    x: FloatArray,
    residual: FloatArray,
) -> FloatArray:
    dimension = x.size
    jacobian = np.empty((dimension, dimension), dtype=np.float64)
    epsilon = np.sqrt(np.finfo(np.float64).eps)
    for column in range(dimension):
        step = epsilon * max(1.0, abs(x[column]))
        shifted = x.copy()
        shifted[column] += step
        jacobian[:, column] = (evaluate(shifted) - residual) / step
    return jacobian


def newton_solve(
    residual: Residual,
    x0: npt.ArrayLike,
    *,
    jacobian: ResidualJacobian | None = None,
    jvp: ResidualJvp | None = None,
    tol: float = 1e-10,
    max_iterations: int = 30,
    line_search_steps: int = 14,
) -> NewtonResult:
    """Damped Newton solve using explicit J, JVP/GMRES, or finite differences."""

    if jacobian is not None and jvp is not None:
        raise ValueError("provide either jacobian or jvp, not both")
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be positive and finite")
    if max_iterations <= 0 or line_search_steps <= 0:
        raise ValueError("iteration limits must be positive")

    x = as_state(x0, name="x0").copy()
    work = WorkCounter()

    def evaluate(value: FloatArray) -> FloatArray:
        result = np.asarray(residual(value.copy()), dtype=np.float64)
        work.residual_evaluations += 1
        if result.shape != x.shape:
            raise ValueError(f"residual returned shape {result.shape}, expected {x.shape}")
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("residual returned NaN or Inf")
        return np.ascontiguousarray(result)

    r = evaluate(x)
    rnorm = float(np.linalg.norm(r, ord=np.inf))
    if rnorm <= tol:
        return NewtonResult(x, True, rnorm, 0, "initial guess satisfies tolerance", work)

    for iteration in range(1, max_iterations + 1):
        try:
            if jacobian is not None:
                matrix = np.asarray(jacobian(x.copy()), dtype=np.float64)
                work.jacobian_evaluations += 1
                if matrix.shape != (x.size, x.size) or not np.all(np.isfinite(matrix)):
                    raise ValueError("jacobian returned an invalid matrix")
                delta = np.linalg.solve(matrix, -r)
            elif jvp is not None:
                def matvec(vector: npt.ArrayLike) -> FloatArray:
                    work.jvp_evaluations += 1
                    value = np.asarray(jvp(x.copy(), np.asarray(vector, dtype=np.float64)), dtype=np.float64)
                    if value.shape != x.shape or not np.all(np.isfinite(value)):
                        raise ValueError("jvp returned an invalid vector")
                    return value
                operator = LinearOperator((x.size, x.size), matvec=matvec, dtype=np.float64)
                def callback(_value: object) -> None:
                    work.krylov_iterations += 1
                delta, info = gmres(
                    operator, -r,
                    rtol=min(1e-8, max(1e-12, 0.05 * rnorm)),
                    atol=0.0,
                    restart=min(20, x.size),
                    maxiter=max(20, 4 * x.size),
                    callback=callback,
                    callback_type="pr_norm",
                )
                if info != 0:
                    return NewtonResult(x, False, rnorm, iteration - 1, f"GMRES failed with info={info}", work)
            else:
                matrix = _finite_difference_jacobian(evaluate, x, r)
                work.jacobian_evaluations += 1
                try:
                    delta = np.linalg.solve(matrix, -r)
                except np.linalg.LinAlgError:
                    delta, *_ = np.linalg.lstsq(matrix, -r, rcond=None)
        except np.linalg.LinAlgError as exc:
            return NewtonResult(x, False, rnorm, iteration - 1, f"linear solve failed: {exc}", work)

        if not np.all(np.isfinite(delta)):
            return NewtonResult(x, False, rnorm, iteration - 1, "non-finite Newton correction", work)

        accepted = False
        alpha = 1.0
        for _ in range(line_search_steps):
            trial = x + alpha * delta
            trial_residual = evaluate(trial)
            trial_norm = float(np.linalg.norm(trial_residual, ord=np.inf))
            if trial_norm <= (1.0 - 1e-4 * alpha) * rnorm or trial_norm <= tol:
                x = trial
                r = trial_residual
                rnorm = trial_norm
                accepted = True
                break
            alpha *= 0.5
        work.newton_iterations += 1
        if not accepted:
            return NewtonResult(x, False, rnorm, iteration, "line search failed", work)
        if rnorm <= tol:
            return NewtonResult(x, True, rnorm, iteration, "converged", work)

    return NewtonResult(x, False, rnorm, max_iterations, "maximum iterations exceeded", work)
