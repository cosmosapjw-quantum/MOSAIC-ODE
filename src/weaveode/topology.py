"""Small-cloud topology primitives used to allocate EESS and homotopy work.

The implementation is intentionally exact and dependency-light for small
candidate clouds. It computes Vietoris--Rips persistence through H1 by reducing
vertices, edges, and triangles over GF(2). Topology is observational: it never
certifies an ODE root.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import inf
from typing import Iterable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _as_points(points: npt.ArrayLike) -> FloatArray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("points must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must be finite")
    return np.ascontiguousarray(array)


def radius_components(points: npt.ArrayLike, radius: float) -> IntArray:
    """Return deterministic labels for Euclidean radius-graph components."""

    array = _as_points(points)
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative")
    try:
        from . import _native
        return np.asarray(_native.radius_components(array, radius), dtype=np.int64)
    except (ImportError, AttributeError):
        pass

    parent = np.arange(array.shape[0], dtype=np.int64)
    rank = np.zeros(array.shape[0], dtype=np.int8)
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value
    def union(a: int, b: int) -> None:
        a = find(a); b = find(b)
        if a == b: return
        if rank[a] < rank[b]: a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]: rank[a] += 1
    radius_sq = radius * radius
    for i in range(array.shape[0]):
        delta = array[i + 1 :] - array[i]
        distances_sq = np.einsum("ij,ij->i", delta, delta)
        for relative in np.flatnonzero(distances_sq <= radius_sq):
            union(i, i + 1 + int(relative))
    root_to_label: dict[int, int] = {}
    labels = np.empty(array.shape[0], dtype=np.int64)
    for i in range(array.shape[0]):
        root = find(i)
        labels[i] = root_to_label.setdefault(root, len(root_to_label))
    return labels


def merge_profile(points: npt.ArrayLike, radii: Iterable[float]) -> list[tuple[float, int]]:
    array = _as_points(points)
    result: list[tuple[float, int]] = []
    for radius in radii:
        value = float(radius)
        labels = radius_components(array, value)
        result.append((value, int(np.unique(labels).size)))
    return result


@dataclass(frozen=True, slots=True)
class _Simplex:
    vertices: tuple[int, ...]
    dimension: int
    filtration: float


def vietoris_rips_persistence(
    points: npt.ArrayLike,
    *,
    max_dimension: int = 1,
    max_radius: float | None = None,
    point_limit: int = 64,
) -> dict[int, list[tuple[float, float]]]:
    array = _as_points(points)
    if max_dimension not in (0, 1):
        raise ValueError("this small-cloud implementation supports max_dimension 0 or 1")
    if point_limit <= 0 or array.shape[0] > point_limit:
        raise ValueError(f"point cloud exceeds point_limit={point_limit}")
    if max_radius is not None and (not np.isfinite(max_radius) or max_radius < 0.0):
        raise ValueError("max_radius must be finite and nonnegative")
    count = array.shape[0]
    deltas = array[:, None, :] - array[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
    cutoff = float(np.max(distances)) if max_radius is None else float(max_radius)
    simplices: list[_Simplex] = [_Simplex((i,), 0, 0.0) for i in range(count)]
    edge_filtration: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(count), 2):
        value = float(distances[i, j])
        if value <= cutoff:
            edge = (i, j)
            edge_filtration[edge] = value
            simplices.append(_Simplex(edge, 1, value))
    if max_dimension >= 1:
        for i, j, k in combinations(range(count), 3):
            edges = ((i, j), (i, k), (j, k))
            if all(edge in edge_filtration for edge in edges):
                value = max(edge_filtration[edge] for edge in edges)
                if value <= cutoff:
                    simplices.append(_Simplex((i, j, k), 2, value))
    simplices.sort(key=lambda simplex: (simplex.filtration, simplex.dimension, simplex.vertices))
    simplex_index = {simplex.vertices: index for index, simplex in enumerate(simplices)}
    reduced_columns: list[set[int]] = []
    pivot_to_column: dict[int, int] = {}
    positive_columns: list[int] = []
    death_of_birth: dict[int, float] = {}
    for column_index, simplex in enumerate(simplices):
        if simplex.dimension == 0:
            column: set[int] = set()
        elif simplex.dimension == 1:
            column = {simplex_index[(simplex.vertices[0],)], simplex_index[(simplex.vertices[1],)]}
        else:
            i, j, k = simplex.vertices
            column = {simplex_index[(i, j)], simplex_index[(i, k)], simplex_index[(j, k)]}
        while column:
            pivot = max(column)
            previous = pivot_to_column.get(pivot)
            if previous is None: break
            column.symmetric_difference_update(reduced_columns[previous])
        reduced_columns.append(column)
        if not column:
            positive_columns.append(column_index)
        else:
            pivot = max(column)
            pivot_to_column[pivot] = column_index
            death_of_birth[pivot] = simplex.filtration
    persistence: dict[int, list[tuple[float, float]]] = {dimension: [] for dimension in range(max_dimension + 1)}
    for birth_index in positive_columns:
        simplex = simplices[birth_index]
        if simplex.dimension <= max_dimension:
            persistence[simplex.dimension].append((simplex.filtration, death_of_birth.get(birth_index, inf)))
    for intervals in persistence.values():
        intervals.sort(key=lambda interval: (interval[0], interval[1]))
    return persistence
