from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ortools.sat.python import cp_model

from .conflicts import ConflictMetrics, analyze_points
from .cp_sat import _multi_point_lines
from .io import Point
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


Rep = tuple[int, int]


def rotate(point: Point, n: int) -> Point:
    row, column = point
    return column, n - 1 - row


def fundamental_domain(n: int) -> list[Rep]:
    if n % 2 == 0:
        return [(row, column) for row in range(n // 2) for column in range(n // 2)]
    return [(row, column) for row in range((n + 1) // 2) for column in range((n - 1) // 2)]


def point_orbit(rep: Rep, n: int) -> tuple[Point, ...]:
    row, column = rep
    if n % 2 and row == column:
        return (rep, (n - 1 - row, n - 1 - column))
    points: list[Point] = []
    point = rep
    for _ in range(4):
        if point not in points:
            points.append(point)
        point = rotate(point, n)
    return tuple(points)


def rot2_fundamental_domain(n: int) -> list[Rep]:
    """Return one representative for every orbit under a half turn."""
    representatives: list[Rep] = []
    for row in range(n):
        for column in range(n):
            point = (row, column)
            opposite = (n - 1 - row, n - 1 - column)
            if point <= opposite:
                representatives.append(point)
    return representatives


def rot2_point_orbit(rep: Rep, n: int) -> tuple[Point, ...]:
    opposite = (n - 1 - rep[0], n - 1 - rep[1])
    return (rep,) if opposite == rep else (rep, opposite)


def validate_rot2_orbit_partition(n: int) -> None:
    counts: dict[Point, int] = {}
    for rep in rot2_fundamental_domain(n):
        for point in rot2_point_orbit(rep, n):
            counts[point] = counts.get(point, 0) + 1
    expected = {(row, column) for row in range(n) for column in range(n)}
    if set(counts) != expected or set(counts.values()) != {1}:
        raise AssertionError("rot2 orbits do not partition the grid")


def validate_orbit_partition(n: int) -> None:
    counts: dict[Point, int] = {}
    for rep in fundamental_domain(n):
        for point in point_orbit(rep, n):
            counts[point] = counts.get(point, 0) + 1
    anti_diagonal = {(row, n - 1 - row) for row in range(n)} if n % 2 else set()
    expected = {(row, column) for row in range(n) for column in range(n)} - anti_diagonal
    if set(counts) != expected or set(counts.values()) != {1}:
        raise AssertionError("fundamental-domain orbits do not partition the intended grid")


@dataclass(frozen=True)
class LazyRound:
    round_index: int
    status: str
    metrics: ConflictMetrics | None
    new_constraints: int
    total_constraints: int
    solve_seconds: float
    points: tuple[Point, ...] | None


@dataclass(frozen=True)
class LazyResult:
    status: str
    points: tuple[Point, ...] | None
    metrics: ConflictMetrics | None
    rounds: int
    constraints: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FullSymmetryResult:
    status: str
    points: tuple[Point, ...] | None
    variables: int
    line_constraints: int
    model_seconds: float
    solve_seconds: float
    conflicts: int
    branches: int


def line_incidence(line_key: tuple[int, int, int], reps: list[Rep], orbits: dict[Rep, tuple[Point, ...]]) -> tuple[tuple[int, int], ...]:
    coefficient_a, coefficient_b, coefficient_c = line_key
    terms: list[tuple[int, int]] = []
    for index, rep in enumerate(reps):
        coefficient = sum(
            coefficient_a * row + coefficient_b * column + coefficient_c == 0
            for row, column in orbits[rep]
        )
        if coefficient:
            terms.append((index, coefficient))
    return tuple(terms)


def lazy_symmetry_search(
    n: int,
    *,
    max_rounds: int,
    time_limit_per_round: float,
    seed: int,
    workers: int,
    initial_constraints: list[tuple[tuple[int, int], ...]] | None = None,
    hint_points: list[Point] | None = None,
    minimize_hint_distance: bool = False,
    callback: Callable[[LazyRound], None] | None = None,
) -> LazyResult:
    validate_orbit_partition(n)
    reps = fundamental_domain(n)
    orbits = {rep: point_orbit(rep, n) for rep in reps}
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"y_{row}_{column}") for row, column in reps]

    for row in range(n):
        terms = []
        for index, rep in enumerate(reps):
            coefficient = sum(point[0] == row for point in orbits[rep])
            if coefficient:
                terms.append(coefficient * variables[index])
        model.add(sum(terms) == 2)
    model.add(sum(len(orbits[rep]) * variables[index] for index, rep in enumerate(reps)) == 2 * n)

    seen_incidence: set[tuple[tuple[int, int], ...]] = set()
    for incidence in initial_constraints or []:
        if incidence in seen_incidence:
            continue
        model.add(sum(coefficient * variables[index] for index, coefficient in incidence) <= 2)
        seen_incidence.add(incidence)

    best_points: tuple[Point, ...] | None = None
    best_metrics: ConflictMetrics | None = None
    hint: list[int] | None = None
    if hint_points is not None:
        selected = set(hint_points)
        hint = []
        expanded: set[Point] = set()
        for rep in reps:
            orbit = orbits[rep]
            chosen = sum(point in selected for point in orbit)
            if chosen not in (0, len(orbit)):
                raise ValueError("symmetry hint is not a union of complete model orbits")
            hint.append(int(chosen > 0))
            if chosen:
                expanded.update(orbit)
        if expanded != selected:
            raise ValueError("symmetry hint contains points outside the represented rct4 domain")
        if minimize_hint_distance:
            distance_terms = [
                1 - variable if value else variable
                for variable, value in zip(variables, hint)
            ]
            model.minimize(sum(distance_terms))
        row_counts = [0] * n
        for row, _ in selected:
            if 0 <= row < n:
                row_counts[row] += 1
        if len(selected) == 2 * n and row_counts == [2] * n:
            best_points = tuple(sorted(selected))
            best_metrics = analyze_points(list(best_points))
            new_constraints = 0
            for violating_line in best_metrics.lines:
                incidence = line_incidence(violating_line.key, reps, orbits)
                if incidence in seen_incidence:
                    continue
                model.add(sum(coefficient * variables[index] for index, coefficient in incidence) <= 2)
                seen_incidence.add(incidence)
                new_constraints += 1
            if callback:
                callback(
                    LazyRound(
                        0,
                        "HINT",
                        best_metrics,
                        new_constraints,
                        len(seen_incidence),
                        0.0,
                        best_points,
                    )
                )
    started = time.perf_counter()

    for round_index in range(1, max_rounds + 1):
        if hint is not None:
            model.clear_hints()
            for variable, value in zip(variables, hint):
                model.add_hint(variable, value)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_per_round)
        solver.parameters.random_seed = int(seed + round_index - 1)
        solver.parameters.num_search_workers = int(workers)
        solve_started = time.perf_counter()
        status_code = solver.solve(model)
        solve_seconds = time.perf_counter() - solve_started
        status = solver.status_name(status_code)
        if status_code not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            event = LazyRound(round_index, status, None, 0, len(seen_incidence), solve_seconds, None)
            if callback:
                callback(event)
            return LazyResult(status, best_points, best_metrics, round_index, len(seen_incidence), time.perf_counter() - started)

        hint = [int(solver.value(variable)) for variable in variables]
        points = tuple(sorted(point for index, rep in enumerate(reps) if hint[index] for point in orbits[rep]))
        metrics = analyze_points(list(points))
        if best_metrics is None or metrics.lexicographic < best_metrics.lexicographic:
            best_points = points
            best_metrics = metrics
        if metrics.triples == 0:
            ok_a, detail_a = verify_a(list(points), n)
            ok_b, detail_b = verify_b(list(points), n)
            if not (ok_a and ok_b):
                raise AssertionError(f"zero-score symmetry candidate failed verification: A={detail_a}; B={detail_b}")
            event = LazyRound(round_index, status, metrics, 0, len(seen_incidence), solve_seconds, points)
            if callback:
                callback(event)
            return LazyResult("VALID", points, metrics, round_index, len(seen_incidence), time.perf_counter() - started)

        new_constraints = 0
        for violating_line in metrics.lines:
            incidence = line_incidence(violating_line.key, reps, orbits)
            if incidence in seen_incidence:
                continue
            if sum(coefficient for _, coefficient in incidence) <= 2:
                raise AssertionError("violating line produced a tautological incidence")
            model.add(sum(coefficient * variables[index] for index, coefficient in incidence) <= 2)
            seen_incidence.add(incidence)
            new_constraints += 1
        event = LazyRound(round_index, status, metrics, new_constraints, len(seen_incidence), solve_seconds, points)
        if callback:
            callback(event)
        if new_constraints == 0:
            raise AssertionError("violations remain but no separating constraint was generated")

    return LazyResult("ROUND_LIMIT", best_points, best_metrics, max_rounds, len(seen_incidence), time.perf_counter() - started)


def lazy_rot2_search(
    n: int,
    *,
    max_rounds: int,
    time_limit_per_round: float,
    seed: int,
    workers: int,
    initial_constraints: list[tuple[tuple[int, int], ...]] | None = None,
    callback: Callable[[LazyRound], None] | None = None,
) -> LazyResult:
    """Lazy exact-cut search in the less restrictive half-turn class."""
    validate_rot2_orbit_partition(n)
    reps = rot2_fundamental_domain(n)
    orbits = {rep: rot2_point_orbit(rep, n) for rep in reps}
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"z_{row}_{column}") for row, column in reps]
    for row in range(n):
        terms = [
            coefficient * variables[index]
            for index, rep in enumerate(reps)
            if (coefficient := sum(point[0] == row for point in orbits[rep]))
        ]
        model.add(sum(terms) == 2)
    for column in range(n):
        terms = [
            coefficient * variables[index]
            for index, rep in enumerate(reps)
            if (coefficient := sum(point[1] == column for point in orbits[rep]))
        ]
        model.add(sum(terms) == 2)
    model.add(sum(len(orbits[rep]) * variables[index] for index, rep in enumerate(reps)) == 2 * n)

    seen_incidence: set[tuple[tuple[int, int], ...]] = set()
    for incidence in initial_constraints or []:
        if incidence in seen_incidence:
            continue
        model.add(sum(coefficient * variables[index] for index, coefficient in incidence) <= 2)
        seen_incidence.add(incidence)
    best_points: tuple[Point, ...] | None = None
    best_metrics: ConflictMetrics | None = None
    hint: list[int] | None = None
    started = time.perf_counter()

    for round_index in range(1, max_rounds + 1):
        if hint is not None:
            model.clear_hints()
            for variable, value in zip(variables, hint):
                model.add_hint(variable, value)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_per_round)
        solver.parameters.random_seed = int(seed + round_index - 1)
        solver.parameters.num_search_workers = int(workers)
        solve_started = time.perf_counter()
        status_code = solver.solve(model)
        solve_seconds = time.perf_counter() - solve_started
        status = solver.status_name(status_code)
        if status_code not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            event = LazyRound(round_index, status, None, 0, len(seen_incidence), solve_seconds, None)
            if callback:
                callback(event)
            return LazyResult(status, best_points, best_metrics, round_index, len(seen_incidence), time.perf_counter() - started)

        hint = [int(solver.value(variable)) for variable in variables]
        points = tuple(sorted(point for index, rep in enumerate(reps) if hint[index] for point in orbits[rep]))
        metrics = analyze_points(list(points))
        if best_metrics is None or metrics.lexicographic < best_metrics.lexicographic:
            best_points = points
            best_metrics = metrics
        if metrics.triples == 0:
            ok_a, detail_a = verify_a(list(points), n)
            ok_b, detail_b = verify_b(list(points), n)
            if not (ok_a and ok_b):
                raise AssertionError(f"zero-score rot2 candidate failed verification: A={detail_a}; B={detail_b}")
            event = LazyRound(round_index, status, metrics, 0, len(seen_incidence), solve_seconds, points)
            if callback:
                callback(event)
            return LazyResult("VALID", points, metrics, round_index, len(seen_incidence), time.perf_counter() - started)

        new_constraints = 0
        for violating_line in metrics.lines:
            incidence = line_incidence(violating_line.key, reps, orbits)
            if incidence in seen_incidence:
                continue
            model.add(sum(coefficient * variables[index] for index, coefficient in incidence) <= 2)
            seen_incidence.add(incidence)
            new_constraints += 1
        event = LazyRound(round_index, status, metrics, new_constraints, len(seen_incidence), solve_seconds, points)
        if callback:
            callback(event)
        if not new_constraints:
            raise AssertionError("rot2 violations remain but no separating cut was generated")

    return LazyResult("ROUND_LIMIT", best_points, best_metrics, max_rounds, len(seen_incidence), time.perf_counter() - started)


def full_symmetry_search(
    n: int,
    *,
    time_limit: float,
    seed: int,
    workers: int,
    hint_points: list[Point] | None = None,
    log_search: bool = False,
) -> FullSymmetryResult:
    validate_orbit_partition(n)
    started_model = time.perf_counter()
    reps = fundamental_domain(n)
    orbits = {rep: point_orbit(rep, n) for rep in reps}
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"y_{row}_{column}") for row, column in reps]
    for row in range(n):
        terms = []
        for index, rep in enumerate(reps):
            coefficient = sum(point[0] == row for point in orbits[rep])
            if coefficient:
                terms.append(coefficient * variables[index])
        model.add(sum(terms) == 2)
    model.add(sum(len(orbits[rep]) * variables[index] for index, rep in enumerate(reps)) == 2 * n)

    rep_for_point = {
        point: index
        for index, rep in enumerate(reps)
        for point in orbits[rep]
    }
    grid = [(row, column) for row in range(n) for column in range(n)]
    line_groups = _multi_point_lines(grid, n)
    seen_incidence: set[tuple[tuple[int, int], ...]] = set()
    for members in line_groups.values():
        counts: dict[int, int] = {}
        for point_index in members:
            rep_index = rep_for_point.get(grid[point_index])
            if rep_index is not None:
                counts[rep_index] = counts.get(rep_index, 0) + 1
        incidence = tuple(sorted(counts.items()))
        if sum(counts.values()) <= 2 or incidence in seen_incidence:
            continue
        seen_incidence.add(incidence)
        model.add(sum(coefficient * variables[index] for index, coefficient in incidence) <= 2)

    if hint_points is not None:
        selected = set(hint_points)
        for index, rep in enumerate(reps):
            orbit = orbits[rep]
            chosen = sum(point in selected for point in orbit)
            if chosen not in (0, len(orbit)):
                raise ValueError("symmetry hint is not a union of complete model orbits")
            model.add_hint(variables[index], int(chosen > 0))
    model_seconds = time.perf_counter() - started_model

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log_search)
    started_solve = time.perf_counter()
    status_code = solver.solve(model)
    solve_seconds = time.perf_counter() - started_solve
    points = None
    if status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        points = tuple(
            sorted(
                point
                for index, rep in enumerate(reps)
                if solver.value(variables[index])
                for point in orbits[rep]
            )
        )
        ok_a, detail_a = verify_a(list(points), n)
        ok_b, detail_b = verify_b(list(points), n)
        if not (ok_a and ok_b):
            raise AssertionError(f"full symmetry candidate failed verification: A={detail_a}; B={detail_b}")
    return FullSymmetryResult(
        solver.status_name(status_code),
        points,
        len(variables),
        len(seen_incidence),
        model_seconds,
        solve_seconds,
        int(solver.num_conflicts),
        int(solver.num_branches),
    )


def full_rot2_search(
    n: int,
    *,
    time_limit: float,
    seed: int,
    workers: int,
    hint_points: list[Point] | None = None,
    log_search: bool = False,
) -> FullSymmetryResult:
    """Solve the exact model restricted only to 180-degree rotational symmetry."""
    validate_rot2_orbit_partition(n)
    started_model = time.perf_counter()
    reps = rot2_fundamental_domain(n)
    orbits = {rep: rot2_point_orbit(rep, n) for rep in reps}
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"z_{row}_{column}") for row, column in reps]

    for row in range(n):
        terms = []
        for index, rep in enumerate(reps):
            coefficient = sum(point[0] == row for point in orbits[rep])
            if coefficient:
                terms.append(coefficient * variables[index])
        model.add(sum(terms) == 2)
    # A half turn maps rows to rows rather than rows to columns, so column
    # equalities are independent here (unlike in the 90-degree rct4 model).
    for column in range(n):
        terms = []
        for index, rep in enumerate(reps):
            coefficient = sum(point[1] == column for point in orbits[rep])
            if coefficient:
                terms.append(coefficient * variables[index])
        model.add(sum(terms) == 2)
    model.add(sum(len(orbits[rep]) * variables[index] for index, rep in enumerate(reps)) == 2 * n)

    rep_for_point = {
        point: index
        for index, rep in enumerate(reps)
        for point in orbits[rep]
    }
    grid = [(row, column) for row in range(n) for column in range(n)]
    line_groups = _multi_point_lines(grid, n)
    seen_incidence: set[tuple[tuple[int, int], ...]] = set()
    for members in line_groups.values():
        counts: dict[int, int] = {}
        for point_index in members:
            rep_index = rep_for_point[grid[point_index]]
            counts[rep_index] = counts.get(rep_index, 0) + 1
        incidence = tuple(sorted(counts.items()))
        if sum(counts.values()) <= 2 or incidence in seen_incidence:
            continue
        seen_incidence.add(incidence)
        model.add(sum(coefficient * variables[index] for index, coefficient in incidence) <= 2)

    if hint_points is not None:
        selected = set(hint_points)
        for index, rep in enumerate(reps):
            orbit = orbits[rep]
            chosen = sum(point in selected for point in orbit)
            if chosen not in (0, len(orbit)):
                raise ValueError("rot2 hint is not a union of complete model orbits")
            model.add_hint(variables[index], int(chosen > 0))
    model_seconds = time.perf_counter() - started_model

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log_search)
    started_solve = time.perf_counter()
    status_code = solver.solve(model)
    solve_seconds = time.perf_counter() - started_solve
    points = None
    if status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        points = tuple(
            sorted(
                point
                for index, rep in enumerate(reps)
                if solver.value(variables[index])
                for point in orbits[rep]
            )
        )
        ok_a, detail_a = verify_a(list(points), n)
        ok_b, detail_b = verify_b(list(points), n)
        if not (ok_a and ok_b):
            raise AssertionError(f"full rot2 candidate failed verification: A={detail_a}; B={detail_b}")
    return FullSymmetryResult(
        solver.status_name(status_code),
        points,
        len(variables),
        len(seen_incidence),
        model_seconds,
        solve_seconds,
        int(solver.num_conflicts),
        int(solver.num_branches),
    )
