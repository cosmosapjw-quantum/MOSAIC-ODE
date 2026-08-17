"""Small deterministic MOSAIC-ODE pre-product benchmark.

This benchmark is intentionally a proof-of-integration artifact, not a performance
claim. CUDA remains source-only unless an executable device environment is present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from weaveode.vector_pipeline import VectorCandidatePipeline


def _linear_vector_case() -> dict[str, Any]:
    matrix = np.diag(np.array([-2.0, -0.5], dtype=np.float64))

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return matrix @ y

    def jac(_t: float, _y: np.ndarray) -> np.ndarray:
        return matrix

    h = 0.1
    pipeline = VectorCandidatePipeline(
        rhs,
        jacobian=jac,
        step_size=h,
        candidate_radius=0.08,
        candidate_count=9,
        eess_points=8,
        component_radius=0.12,
        atol=np.array([1e-10, 1e-10]),
        rtol=1e-8,
        certification_tol=1e-6,
        seed=20260817,
    )
    y = np.array([1.0, 2.0], dtype=np.float64)
    discrete = y.copy()
    inverse = np.linalg.inv(np.eye(2) - h * matrix)
    errors: list[float] = []
    work = None
    for index in range(6):
        result = pipeline.step(index * h, y)
        if not result.accepted:
            return {"success": False, "message": result.message}
        y = result.y
        discrete = inverse @ discrete
        errors.append(float(np.max(np.abs(y - discrete))))
        work = result.work.to_dict()
    return {
        "success": True,
        "final_state": y.tolist(),
        "max_discrete_error": max(errors, default=0.0),
        "last_step_work": work,
    }


def _robertson_case() -> dict[str, Any]:
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        y1, y2, y3 = y
        return np.array(
            [
                -0.04 * y1 + 1.0e4 * y2 * y3,
                0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
                3.0e7 * y2 * y2,
            ],
            dtype=np.float64,
        )

    def jac(_t: float, y: np.ndarray) -> np.ndarray:
        _y1, y2, y3 = y
        return np.array(
            [
                [-0.04, 1.0e4 * y3, 1.0e4 * y2],
                [0.04, -1.0e4 * y3 - 6.0e7 * y2, -1.0e4 * y2],
                [0.0, 6.0e7 * y2, 0.0],
            ],
            dtype=np.float64,
        )

    pipeline = VectorCandidatePipeline(
        rhs,
        jacobian=jac,
        step_size=1e-4,
        candidate_radius=0.01,
        candidate_count=9,
        eess_points=8,
        component_radius=0.05,
        atol=np.array([1e-12, 1e-14, 1e-12]),
        rtol=1e-8,
        certification_tol=1e-5,
        seed=20260817,
    )
    initial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    result = pipeline.step(0.0, initial)
    state = result.y
    return {
        "accepted": result.accepted,
        "scaled_residual": result.scaled_residual,
        "state": state.tolist(),
        "minimum_state": float(np.min(state)),
        "mass_error": float(abs(np.sum(state) - 1.0)),
        "component_count": result.trace.component_count,
        "homotopy_paths": result.trace.homotopy_paths,
        "selected_source": result.trace.selected_source,
        "work": result.work.to_dict(),
    }


def run_preproduct_benchmark() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project": "MOSAIC-ODE",
        "release_stage": "integrated-pre-product",
        "linear_vector": _linear_vector_case(),
        "robertson": _robertson_case(),
        "cuda": {
            "executed": False,
            "claim": "CUDA source plane is packaged but not executed in this benchmark.",
        },
    }


def main() -> None:
    result = run_preproduct_benchmark()
    target = Path("artifacts/generated/preproduct_benchmark.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(target)


if __name__ == "__main__":
    main()
