"""Small analytic problems used by the integrated coding research loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PolynomialScalarProblem:
    """Autonomous scalar ODE ``y'=sum_k c_k y^k``."""

    coefficients: FloatArray

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.ndim != 1 or coefficients.size == 0 or not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must be a non-empty finite vector")
        object.__setattr__(self, "coefficients", np.ascontiguousarray(coefficients))

    def rhs(self, _t: float, y: FloatArray) -> FloatArray:
        state = np.asarray(y, dtype=np.float64)
        if state.shape != (1,):
            raise ValueError("PolynomialScalarProblem expects one state component")
        value = 0.0
        for coefficient in self.coefficients[::-1]:
            value = value * state[0] + float(coefficient)
        return np.array([value], dtype=np.float64)

    def jacobian(self, _t: float, y: FloatArray) -> FloatArray:
        state = np.asarray(y, dtype=np.float64)
        if state.shape != (1,):
            raise ValueError("PolynomialScalarProblem expects one state component")
        derivative = 0.0
        for power in range(1, self.coefficients.size):
            derivative += power * self.coefficients[power] * state[0] ** (power - 1)
        return np.array([[derivative]], dtype=np.float64)
