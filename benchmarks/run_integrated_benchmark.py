"""Fixed-seed synergy ladder for the integrated WeaveODE vertical slice."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np

from weaveode import _native
from weaveode.problems import PolynomialScalarProblem
from weaveode.solver import WeaveBDF1Solver
from weaveode.telemetry import WorkCounter
from weaveode.topology import radius_components
from weaveode.trajroot import solve_bdf1_step


def _principal_quadratic_root(previous: float, h: float) -> float:
    discriminant = 1.0 - 4.0 * h * previous
    if discriminant <= 0.0:
        raise RuntimeError("benchmark left the real principal BDF1 branch")
    return 2.0 * previous / (1.0 + math.sqrt(discriminant))


def _work_units(work: WorkCounter) -> int:
    return int(
        work.residual_evaluations
        + work.rhs_evaluations
        + 2 * work.jacobian_evaluations
        + work.jvp_evaluations
        + work.newton_iterations
        + work.krylov_iterations
        + work.native_candidate_evaluations
        + work.topology_evaluations
        + work.eess_candidate_evaluations
        + 4 * work.online_updates
        + 4 * work.fallbacks
    )


def _add_work(total: WorkCounter, addition: WorkCounter) -> None:
    total += addition


def _score_grid(
    grid: np.ndarray,
    previous: float,
    h: float,
    coefficients: np.ndarray,
    work: WorkCounter,
    *,
    atol: float,
    rtol: float,
) -> np.ndarray:
    scores = np.asarray(
        _native.poly_bdf1_scores(grid, previous, h, coefficients, atol, rtol),
        dtype=np.float64,
    )
    work.native_candidate_evaluations += grid.size
    return scores


def _refine_and_select(
    problem: PolynomialScalarProblem,
    previous: float,
    h: float,
    seeds: np.ndarray,
    predictor: float,
    work: WorkCounter,
    *,
    atol: float,
    rtol: float,
    certification_tol: float,
) -> tuple[float, float]:
    roots: list[tuple[float, float]] = []
    for seed in seeds:
        result = solve_bdf1_step(
            problem.rhs,
            0.0,
            np.array([previous], dtype=np.float64),
            h,
            predictor=np.array([float(seed)], dtype=np.float64),
            jacobian=problem.jacobian,
            atol=atol,
            rtol=rtol,
            certification_tol=certification_tol,
        )
        _add_work(work, result.work)
        if result.accepted:
            root = float(result.y[0])
            if not any(abs(root - existing[0]) < 1e-10 for existing in roots):
                roots.append((root, result.scaled_residual))
    if not roots:
        raise RuntimeError("candidate refinement produced no certified root")
    return min(roots, key=lambda pair: (abs(pair[0] - predictor), pair[1]))


def _run_backbone(
    problem: PolynomialScalarProblem,
    y0: float,
    h: float,
    steps: int,
    references: list[float],
    *,
    atol: float,
    rtol: float,
    certification_tol: float,
) -> dict[str, Any]:
    work = WorkCounter()
    current = y0
    errors: list[float] = []
    residuals: list[float] = []
    started = time.perf_counter()
    for index in range(steps):
        result = solve_bdf1_step(
            problem.rhs,
            index * h,
            np.array([current], dtype=np.float64),
            h,
            jacobian=problem.jacobian,
            atol=atol,
            rtol=rtol,
            certification_tol=certification_tol,
        )
        _add_work(work, result.work)
        if not result.accepted:
            return {"success": False, "message": result.message}
        current = float(result.y[0])
        errors.append(abs(current - references[index + 1]))
        residuals.append(result.scaled_residual)
    elapsed = time.perf_counter() - started
    return _result_record(current, errors, residuals, work, elapsed)


def _run_search_variant(
    problem: PolynomialScalarProblem,
    y0: float,
    h: float,
    steps: int,
    references: list[float],
    *,
    topology: bool,
    atol: float,
    rtol: float,
    certification_tol: float,
) -> dict[str, Any]:
    work = WorkCounter()
    current = y0
    errors: list[float] = []
    residuals: list[float] = []
    component_counts: list[int] = []
    started = time.perf_counter()
    for index in range(steps):
        predictor = current + h * float(problem.rhs(index * h, np.array([current]))[0])
        work.rhs_evaluations += 1
        grid = np.linspace(predictor - 14.0, predictor + 14.0, 129, dtype=np.float64)
        scores = _score_grid(
            grid, current, h, problem.coefficients, work, atol=atol, rtol=rtol
        )
        selected_indices = np.asarray(np.argsort(scores, kind="stable")[:16], dtype=np.int64)
        selected_grid = grid[selected_indices]
        selected_scores = scores[selected_indices]
        work.eess_candidate_evaluations += selected_grid.size

        if topology:
            labels = radius_components(selected_grid[:, None], 0.65)
            work.topology_evaluations += 1
            component_counts.append(int(np.unique(labels).size))
            representatives: list[float] = []
            for label in np.unique(labels):
                members = np.flatnonzero(labels == label)
                representatives.append(float(selected_grid[members[np.argmin(selected_scores[members])]]))
            seeds = np.asarray(representatives, dtype=np.float64)
        else:
            component_counts.append(0)
            seeds = selected_grid[np.argsort(selected_scores)[:8]]

        current, score = _refine_and_select(
            problem,
            current,
            h,
            seeds,
            predictor,
            work,
            atol=atol,
            rtol=rtol,
            certification_tol=certification_tol,
        )
        errors.append(abs(current - references[index + 1]))
        residuals.append(score)
    elapsed = time.perf_counter() - started
    record = _result_record(current, errors, residuals, work, elapsed)
    record["component_counts"] = component_counts
    return record


def _run_integrated(
    problem: PolynomialScalarProblem,
    y0: float,
    h: float,
    steps: int,
    references: list[float],
    *,
    seed: int,
    atol: float,
    rtol: float,
    certification_tol: float,
) -> dict[str, Any]:
    solver = WeaveBDF1Solver(
        problem.rhs,
        polynomial_coefficients=problem.coefficients,
        jacobian=problem.jacobian,
        step_size=h,
        atol=atol,
        rtol=rtol,
        certification_tol=certification_tol,
        search_radius=14.0,
        grid_candidates=129,
        eess_points=16,
        component_radius=0.65,
        seed=seed,
    )
    current = np.array([y0], dtype=np.float64)
    errors: list[float] = []
    residuals: list[float] = []
    components: list[int] = []
    total = WorkCounter()
    started = time.perf_counter()
    for index in range(steps):
        result = solver.step(index * h, current)
        _add_work(total, result.work)
        if not result.accepted:
            return {"success": False, "message": result.message}
        current = result.y
        errors.append(abs(float(current[0]) - references[index + 1]))
        residuals.append(result.scaled_residual)
        components.append(result.trace.component_count)
    elapsed = time.perf_counter() - started
    record = _result_record(float(current[0]), errors, residuals, total, elapsed)
    record["component_counts"] = components
    record["online_updates"] = total.online_updates
    record["fallbacks"] = total.fallbacks
    return record


def _result_record(
    final_state: float,
    errors: list[float],
    residuals: list[float],
    work: WorkCounter,
    elapsed: float,
) -> dict[str, Any]:
    max_error = max(errors, default=0.0)
    max_residual = max(residuals, default=0.0)
    return {
        "success": True,
        "final_state": final_state,
        "max_principal_branch_error": max_error,
        "max_scaled_residual": max_residual,
        "false_acceptances": int(sum(error > 1e-8 for error in errors)),
        "work_units": _work_units(work),
        "work": work.to_dict(),
        "wall_seconds": elapsed,
    }


def run_benchmark(*, seed: int = 20260817, steps: int = 6) -> dict[str, Any]:
    np.random.seed(seed)
    h = 0.08
    y0 = 0.10
    atol = 1e-10
    rtol = 1e-8
    certification_tol = 1e-6
    problem = PolynomialScalarProblem(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    references = [y0]
    for _ in range(steps):
        references.append(_principal_quadratic_root(references[-1], h))

    variants = {
        "B": _run_backbone(
            problem, y0, h, steps, references,
            atol=atol, rtol=rtol, certification_tol=certification_tol,
        ),
        "B+E": _run_search_variant(
            problem, y0, h, steps, references, topology=False,
            atol=atol, rtol=rtol, certification_tol=certification_tol,
        ),
        "B+T": _run_search_variant(
            problem, y0, h, steps, references, topology=True,
            atol=atol, rtol=rtol, certification_tol=certification_tol,
        ),
        "B+E+T+H+M": _run_integrated(
            problem, y0, h, steps, references, seed=seed,
            atol=atol, rtol=rtol, certification_tol=certification_tol,
        ),
    }
    costs = {name: max(1, int(value["work_units"])) for name, value in variants.items()}
    synergy = {
        "integrated_vs_additive_log_work": (
            -math.log(costs["B+E+T+H+M"])
            + math.log(costs["B+E"])
            + math.log(costs["B+T"])
            - math.log(costs["B"])
        ),
        "integrated_work_ratio_to_backbone": costs["B+E+T+H+M"] / costs["B"],
        "interpretation": "Descriptive factorial-style interaction metric; positive is lower-than-additive work. No gain is assumed.",
    }
    return {
        "schema_version": 1,
        "seed": seed,
        "problem": {
            "name": "quadratic_growth_bdf1_multibranch",
            "equation": "y'=y^2",
            "y0": y0,
            "step_size": h,
            "steps": steps,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "variants": variants,
        "synergy": synergy,
        "cuda": {
            "source_present": (Path(__file__).parents[1] / "cuda" / "candidate_score.cu").exists(),
            "executed": False,
            "reason": "CUDA device/toolchain unavailable in the validation environment",
        },
    }


def deterministic_projection(result: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(result)
    projected.pop("environment", None)
    for variant in projected.get("variants", {}).values():
        variant.pop("wall_seconds", None)
    return projected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated/integrated_benchmark.json"))
    args = parser.parse_args()
    result = run_benchmark(seed=args.seed, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
