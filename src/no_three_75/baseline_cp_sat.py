from __future__ import annotations

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .cp_sat import _multi_point_lines
from .io import Point
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class BaselineResult:
    status: str
    points: tuple[Point, ...] | None
    variables: int
    line_constraints: int
    model_seconds: float
    solve_seconds: float
    conflicts: int
    branches: int


def solve_baseline(
    n: int,
    *,
    time_limit: float,
    seed: int,
    workers: int,
    log_search: bool = False,
    hint_points: list[Point] | None = None,
) -> BaselineResult:
    started_model = time.perf_counter()
    model = cp_model.CpModel()
    points = [(row, column) for row in range(n) for column in range(n)]
    variables = [model.new_bool_var(f"x_{row}_{column}") for row, column in points]
    for row in range(n):
        model.add(sum(variables[row * n + column] for column in range(n)) == 2)
    for column in range(n):
        model.add(sum(variables[row * n + column] for row in range(n)) == 2)

    line_groups = _multi_point_lines(points, n)
    for members in line_groups.values():
        model.add(sum(variables[index] for index in members) <= 2)
    if hint_points is not None:
        selected_hint = set(hint_points)
        for point, variable in zip(points, variables):
            model.add_hint(variable, 1 if point in selected_hint else 0)
    model_seconds = time.perf_counter() - started_model

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log_search)
    started_solve = time.perf_counter()
    status_code = solver.solve(model)
    solve_seconds = time.perf_counter() - started_solve
    selected = None
    if status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        selected = tuple(points[index] for index, variable in enumerate(variables) if solver.value(variable))
        ok_a, detail_a = verify_a(list(selected), n)
        ok_b, detail_b = verify_b(list(selected), n)
        if not (ok_a and ok_b):
            raise AssertionError(f"baseline CP-SAT witness failed verification: A={detail_a}; B={detail_b}")
    return BaselineResult(
        solver.status_name(status_code),
        selected,
        len(variables),
        len(line_groups),
        model_seconds,
        solve_seconds,
        int(solver.num_conflicts),
        int(solver.num_branches),
    )
