from __future__ import annotations

import time
from dataclasses import dataclass
from math import gcd
from typing import Iterable

from ortools.sat.python import cp_model

from .conflicts import ConflictMetrics, analyze_points
from .io import Point
from .state import PairPermutations, points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class RepairResult:
    status: str
    state: PairPermutations | None
    free_rows: tuple[int, ...]
    variables: int
    line_constraints: int
    model_seconds: float
    solve_seconds: float
    conflicts: int
    branches: int


def greedy_conflict_row_cover(metrics: ConflictMetrics) -> list[int]:
    line_rows = [set(row for row, _ in line.points) for line in metrics.lines]
    uncovered = set(range(len(line_rows)))
    chosen: list[int] = []
    while uncovered:
        counts: dict[int, int] = {}
        for line_index in uncovered:
            for row in line_rows[line_index]:
                counts[row] = counts.get(row, 0) + 1
        row = min(counts, key=lambda candidate: (-counts[candidate], candidate))
        chosen.append(row)
        uncovered = {line_index for line_index in uncovered if row not in line_rows[line_index]}
    return chosen


def randomized_conflict_row_cover(metrics: ConflictMetrics, seed: int) -> list[int]:
    """Return a reproducible, diversified hitting set of conflict lines.

    The deterministic greedy cover is useful for making a small repair model,
    but repeatedly fixing the same cover leaves a large part of the LNS
    portfolio identical.  This variant chooses among rows in proportion to the
    square of their uncovered-line coverage.  Every returned cover is still
    checked constructively: each violating line contributes a row before the
    routine stops.
    """
    import random

    rng = random.Random(seed)
    line_rows = [set(row for row, _ in line.points) for line in metrics.lines]
    uncovered = set(range(len(line_rows)))
    chosen: list[int] = []
    while uncovered:
        counts: dict[int, int] = {}
        for line_index in uncovered:
            for row in line_rows[line_index]:
                counts[row] = counts.get(row, 0) + 1
        candidates = sorted(counts)
        weights = [counts[row] * counts[row] for row in candidates]
        row = rng.choices(candidates, weights=weights, k=1)[0]
        chosen.append(row)
        uncovered = {line_index for line_index in uncovered if row not in line_rows[line_index]}
    return chosen


def choose_free_rows(
    state: PairPermutations,
    target_count: int,
    seed: int = 0,
    *,
    strategy: str = "all-conflicts",
) -> list[int]:
    import random

    metrics = analyze_points(state.points())
    if strategy == "all-conflicts":
        chosen = greedy_conflict_row_cover(metrics)
    elif strategy == "cover-random":
        chosen = randomized_conflict_row_cover(metrics, seed)
    else:
        raise ValueError(f"unknown free-row strategy {strategy}")
    if len(chosen) > target_count:
        raise ValueError(f"at least {len(chosen)} rows are needed to hit every current conflict")
    conflict_row_weight: dict[int, int] = {}
    for line in metrics.lines:
        for row, _ in line.points:
            conflict_row_weight[row] = conflict_row_weight.get(row, 0) + line.triples
    rng = random.Random(seed)
    if strategy == "all-conflicts":
        for row in sorted(conflict_row_weight, key=lambda candidate: (-conflict_row_weight[candidate], candidate)):
            if len(chosen) >= target_count:
                break
            if row not in chosen:
                chosen.append(row)
    remaining = [row for row in range(state.n) if row not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: target_count - len(chosen)])
    return sorted(chosen)


def _packed_line(a: Point, b: Point, n: int) -> int:
    row_a, column_a = a
    row_b, column_b = b
    coefficient_a = column_b - column_a
    coefficient_b = row_a - row_b
    coefficient_c = -(coefficient_a * row_a + coefficient_b * column_a)
    divisor = gcd(gcd(abs(coefficient_a), abs(coefficient_b)), abs(coefficient_c))
    coefficient_a //= divisor
    coefficient_b //= divisor
    coefficient_c //= divisor
    if coefficient_a < 0 or (coefficient_a == 0 and coefficient_b < 0):
        coefficient_a = -coefficient_a
        coefficient_b = -coefficient_b
        coefficient_c = -coefficient_c
    coefficient_span = 2 * n - 1
    c_offset = 2 * (n - 1) * (n - 1)
    c_span = 2 * c_offset + 1
    return ((coefficient_a + n - 1) * coefficient_span + coefficient_b + n - 1) * c_span + coefficient_c + c_offset


def _multi_point_lines(active: list[Point], n: int) -> dict[int, set[int]]:
    first_pair: dict[int, int] = {}
    multiple: dict[int, set[int]] = {}
    index_bits = max(1, (len(active) - 1).bit_length())
    index_mask = (1 << index_bits) - 1
    for left in range(len(active) - 1):
        left_point = active[left]
        for right in range(left + 1, len(active)):
            right_point = active[right]
            # Row/column equalities already enforce their at-most-two constraints.
            if left_point[0] == right_point[0] or left_point[1] == right_point[1]:
                continue
            key = _packed_line(left_point, right_point, n)
            members = multiple.get(key)
            if members is not None:
                members.add(left)
                members.add(right)
                continue
            packed_pair = first_pair.get(key)
            if packed_pair is None:
                # +1 distinguishes a packed (0,0) value from a missing dictionary entry.
                first_pair[key] = ((left << index_bits) | right) + 1
                continue
            packed_pair -= 1
            first_left = packed_pair >> index_bits
            first_right = packed_pair & index_mask
            multiple[key] = {first_left, first_right, left, right}
            del first_pair[key]
    return multiple


def repair_state(
    state: PairPermutations,
    free_rows: Iterable[int],
    *,
    time_limit: float = 30.0,
    seed: int = 0,
    workers: int = 1,
    log_search: bool = False,
) -> RepairResult:
    state.validate()
    n = state.n
    free = tuple(sorted(set(int(row) for row in free_rows)))
    if not free or any(row < 0 or row >= n for row in free):
        raise ValueError("free rows must be a nonempty in-grid set")
    free_set = set(free)
    started_model = time.perf_counter()
    model = cp_model.CpModel()
    variables = {
        (row, column): model.new_bool_var(f"x_{row}_{column}")
        for row in free
        for column in range(n)
    }
    fixed = [point for point in state.points() if point[0] not in free_set]

    for row in free:
        model.add(sum(variables[row, column] for column in range(n)) == 2)
    fixed_column_counts = [0] * n
    for _, column in fixed:
        fixed_column_counts[column] += 1
    for column in range(n):
        needed = 2 - fixed_column_counts[column]
        if needed < 0:
            raise ValueError("fixed rows already overfill a column")
        model.add(sum(variables[row, column] for row in free) == needed)

    active = fixed + [(row, column) for row in free for column in range(n)]
    fixed_count = len(fixed)
    line_groups = _multi_point_lines(active, n)
    line_constraints = 0
    for members in line_groups.values():
        fixed_on_line = sum(index < fixed_count for index in members)
        if fixed_on_line >= 3:
            # The caller did not free a hitting set for every existing conflict.
            model.add_bool_or([])
            line_constraints += 1
            continue
        variable_points = [active[index] for index in members if index >= fixed_count]
        bound = 2 - fixed_on_line
        if len(variable_points) > bound:
            model.add(sum(variables[point] for point in variable_points) <= bound)
            line_constraints += 1

    current = set(state.points())
    for point, variable in variables.items():
        model.add_hint(variable, 1 if point in current else 0)
    model_seconds = time.perf_counter() - started_model

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log_search)
    started_solve = time.perf_counter()
    status_code = solver.solve(model)
    solve_seconds = time.perf_counter() - started_solve
    status = solver.status_name(status_code)
    repaired_state = None
    if status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        points = list(fixed)
        points.extend(point for point, variable in variables.items() if solver.value(variable))
        ok_a, detail_a = verify_a(points, n)
        ok_b, detail_b = verify_b(points, n)
        if not (ok_a and ok_b):
            raise AssertionError(f"CP-SAT returned an invalid repair: A={detail_a}; B={detail_b}")
        repaired_state = points_to_permutations(points, n)
    return RepairResult(
        status,
        repaired_state,
        free,
        len(variables),
        line_constraints,
        model_seconds,
        solve_seconds,
        int(solver.num_conflicts),
        int(solver.num_branches),
    )
