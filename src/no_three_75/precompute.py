from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from math import comb, isqrt

import numpy as np

from .accelerated import score_and_badness
from .conflicts import analyze_points
from .cp_sat import _multi_point_lines
from .io import Point
from .state import points_to_permutations


@dataclass(frozen=True)
class GeometryTables:
    n: int
    pair_to_line: np.ndarray
    offsets: np.ndarray
    indices: np.ndarray
    line_count: int
    build_seconds: float


def triangular_pair_index(left: int, right: int, point_count: int) -> int:
    if left > right:
        left, right = right, left
    if left == right or left < 0 or right >= point_count:
        raise ValueError("pair index requires two distinct in-range points")
    return left * (2 * point_count - left - 1) // 2 + right - left - 1


def build_geometry_tables(n: int) -> GeometryTables:
    started = time.perf_counter()
    grid = [(row, column) for row in range(n) for column in range(n)]
    groups = _multi_point_lines(grid, n)
    point_count = n * n
    pair_to_line = np.full(point_count * (point_count - 1) // 2, -1, dtype=np.int32)
    offsets = np.empty(len(groups) + 1, dtype=np.int64)
    offsets[0] = 0
    flat_members: list[int] = []
    for line_id, members_set in enumerate(groups.values()):
        members = sorted(members_set)
        flat_members.extend(members)
        offsets[line_id + 1] = len(flat_members)
        for left, right in combinations(members, 2):
            index = triangular_pair_index(left, right, point_count)
            previous = int(pair_to_line[index])
            if previous not in (-1, line_id):
                raise AssertionError("a grid-point pair was assigned to two geometric lines")
            pair_to_line[index] = line_id
    indices = np.asarray(flat_members, dtype=np.int32)
    return GeometryTables(n, pair_to_line, offsets, indices, len(groups), time.perf_counter() - started)


def score_pair_table(points: list[Point], tables: GeometryTables) -> int:
    ids = sorted(row * tables.n + column for row, column in points)
    point_count = tables.n * tables.n
    pair_indices = np.fromiter(
        (triangular_pair_index(left, right, point_count) for left, right in combinations(ids, 2)),
        dtype=np.int64,
    )
    line_ids = tables.pair_to_line[pair_indices]
    line_ids = line_ids[line_ids >= 0]
    if len(line_ids) == 0:
        return 0
    _, pair_counts = np.unique(line_ids, return_counts=True)
    score = 0
    for pair_count in pair_counts:
        discriminant = 1 + 8 * int(pair_count)
        root = isqrt(discriminant)
        if root * root != discriminant:
            raise AssertionError("selected pair count is not triangular")
        occupancy = (1 + root) // 2
        if occupancy >= 3:
            score += comb(occupancy, 3)
    return score


def score_csr(points: list[Point], tables: GeometryTables) -> int:
    occupied = np.zeros(tables.n * tables.n, dtype=np.int8)
    occupied[[row * tables.n + column for row, column in points]] = 1
    counts = np.add.reduceat(occupied[tables.indices], tables.offsets[:-1])
    violating = counts[counts >= 3].astype(np.int64)
    return int(np.sum(violating * (violating - 1) * (violating - 2) // 6))


def _average_seconds(function, repeats: int) -> tuple[int, float]:
    started = time.perf_counter()
    value = None
    for _ in range(repeats):
        value = int(function())
    return int(value), (time.perf_counter() - started) / repeats


def benchmark_precomputation(n: int, points: list[Point]) -> dict[str, object]:
    state = points_to_permutations(points, n)
    p = np.asarray(state.P, dtype=np.int32)
    q = np.asarray(state.Q, dtype=np.int32)
    score_and_badness(p, q)
    tables = build_geometry_tables(n)
    score_a, seconds_a = _average_seconds(lambda: score_and_badness(p, q)[0], 100)
    score_b, seconds_b = _average_seconds(lambda: score_pair_table(points, tables), 100)
    score_c, seconds_c = _average_seconds(lambda: analyze_points(points).triples, 5)
    score_d, seconds_d = _average_seconds(lambda: score_csr(points, tables), 20)
    if len({score_a, score_b, score_c, score_d}) != 1:
        raise AssertionError(f"precompute scorers disagree: {score_a}, {score_b}, {score_c}, {score_d}")
    return {
        "n": n,
        "score": score_a,
        "line_count": tables.line_count,
        "line_incidences": int(len(tables.indices)),
        "build_seconds": tables.build_seconds,
        "memory_bytes": {
            "pair_to_line_triangular": int(tables.pair_to_line.nbytes),
            "csr_offsets": int(tables.offsets.nbytes),
            "csr_indices": int(tables.indices.nbytes),
        },
        "seconds_per_score": {
            "A_numba_determinants": seconds_a,
            "B_pair_to_line": seconds_b,
            "C_canonical_line_dict": seconds_c,
            "D_csr_line_scan": seconds_d,
        },
    }
