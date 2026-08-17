"""Small, explicit numerical contracts for the integrated research slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt

from .telemetry import WorkCounter

FloatArray = npt.NDArray[np.float64]
Rhs = Callable[[float, FloatArray], FloatArray]
Jacobian = Callable[[float, FloatArray], FloatArray]


class MatrixGenerator(Protocol):
    def __call__(self, t: float) -> FloatArray: ...


@dataclass(slots=True)
class IntegrationResult:
    t: FloatArray
    y: FloatArray
    success: bool
    message: str
    work: WorkCounter


def as_state(value: npt.ArrayLike, *, name: str = "state") -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(array)


def positive_step(h: float) -> float:
    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("step size must be positive and finite")
    return h


def tolerance_scale(
    reference: FloatArray,
    candidate: FloatArray,
    atol: float | npt.ArrayLike,
    rtol: float,
) -> FloatArray:
    if rtol < 0.0 or not np.isfinite(rtol):
        raise ValueError("rtol must be finite and nonnegative")
    atol_array = np.asarray(atol, dtype=np.float64)
    if np.any(~np.isfinite(atol_array)) or np.any(atol_array < 0.0):
        raise ValueError("atol must be finite and nonnegative")
    try:
        scale = atol_array + rtol * np.maximum(np.abs(reference), np.abs(candidate))
    except ValueError as exc:
        raise ValueError("atol is not broadcast-compatible with the state") from exc
    if np.any(scale <= 0.0):
        raise ValueError("atol and rtol cannot both produce a zero scale")
    return np.asarray(scale, dtype=np.float64)


def wrms_norm(values: FloatArray, scale: FloatArray) -> float:
    ratio = np.asarray(values, dtype=np.float64) / np.asarray(scale, dtype=np.float64)
    return float(np.sqrt(np.mean(ratio * ratio)))
