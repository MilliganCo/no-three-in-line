from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterable

from ortools.sat.python import cp_model

from .conflicts import analyze_points
from .cp_sat import _multi_point_lines
from .io import Point
from .state import PairPermutations, points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class EdgeRepairResult:
    status: str
    state: PairPermutations | None
    free_points: tuple[Point, ...]
    active_rows: int
    active_columns: int
    variables: int
    line_constraints: int
    model_seconds: float
    solve_seconds: float
    conflicts: int
    branches: int


def choose_free_points(state: PairPermutations, target_count: int, *, seed: int) -> list[Point]:
    """Free every conflicting point, then a reproducible spread of extra edges."""
    points = state.points()
    metrics = analyze_points(points)
    required = {point for line in metrics.lines for point in line.points}
    if target_count < len(required) or target_count > len(points):
        raise ValueError(f"target_count must be in [{len(required)}, {len(points)}]")

    # Partners in conflict rows are useful first: they recover the complete
    # row-LNS neighborhood while retaining the option to spread later edges.
    conflict_rows = {row for row, _ in required}
    partners = [point for point in points if point[0] in conflict_rows and point not in required]
    rng = random.Random(seed)
    rng.shuffle(partners)
    remaining = [point for point in points if point not in required and point not in partners]
    rng.shuffle(remaining)
    selected = sorted(required)
    selected.extend(partners[: max(0, min(len(partners), target_count - len(selected)))])
    selected_set = set(selected)
    selected.extend(point for point in remaining if point not in selected_set)
    return sorted(selected[:target_count])


def repair_state_edges(
    state: PairPermutations,
    free_points: Iterable[Point],
    *,
    time_limit: float,
    seed: int,
    workers: int,
    log_search: bool = False,
) -> EdgeRepairResult:
    """Exact LNS with individual selected edges, rather than whole rows, unfixed."""
    state.validate()
    n = state.n
    selected = set(state.points())
    free = tuple(sorted(set(free_points)))
    if not free or any(point not in selected for point in free):
        raise ValueError("free_points must be a nonempty subset of the current state")
    fixed = sorted(selected - set(free))

    fixed_rows = [0] * n
    fixed_columns = [0] * n
    for row, column in fixed:
        fixed_rows[row] += 1
        fixed_columns[column] += 1
    row_need = [2 - count for count in fixed_rows]
    column_need = [2 - count for count in fixed_columns]
    if min(row_need) < 0 or min(column_need) < 0 or sum(row_need) != sum(column_need):
        raise AssertionError("invalid residual degrees")
    active_rows = [row for row, need in enumerate(row_need) if need]
    active_columns = [column for column, need in enumerate(column_need) if need]

    started_model = time.perf_counter()
    model = cp_model.CpModel()
    variables = {
        (row, column): model.new_bool_var(f"e_{row}_{column}")
        for row in active_rows
        for column in active_columns
        if (row, column) not in fixed
    }
    for row in active_rows:
        model.add(sum(variable for (r, _), variable in variables.items() if r == row) == row_need[row])
    for column in active_columns:
        model.add(sum(variable for (_, c), variable in variables.items() if c == column) == column_need[column])

    active = fixed + list(variables)
    fixed_count = len(fixed)
    line_constraints = 0
    for members in _multi_point_lines(active, n).values():
        fixed_on_line = sum(index < fixed_count for index in members)
        if fixed_on_line >= 3:
            model.add_bool_or([])
            line_constraints += 1
            continue
        variable_points = [active[index] for index in members if index >= fixed_count]
        bound = 2 - fixed_on_line
        if len(variable_points) > bound:
            model.add(sum(variables[point] for point in variable_points) <= bound)
            line_constraints += 1

    for point, variable in variables.items():
        model.add_hint(variable, int(point in selected))
    model_seconds = time.perf_counter() - started_model

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log_search)
    started_solve = time.perf_counter()
    status_code = solver.solve(model)
    solve_seconds = time.perf_counter() - started_solve
    repaired = None
    if status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        points = fixed + [point for point, variable in variables.items() if solver.value(variable)]
        ok_a, detail_a = verify_a(points, n)
        ok_b, detail_b = verify_b(points, n)
        if not (ok_a and ok_b):
            raise AssertionError(f"edge LNS returned invalid state: A={detail_a}; B={detail_b}")
        repaired = points_to_permutations(points, n)
    return EdgeRepairResult(
        solver.status_name(status_code),
        repaired,
        free,
        len(active_rows),
        len(active_columns),
        len(variables),
        line_constraints,
        model_seconds,
        solve_seconds,
        int(solver.num_conflicts),
        int(solver.num_branches),
    )
