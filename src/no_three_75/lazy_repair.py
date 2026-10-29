from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

from ortools.sat.python import cp_model

from .conflicts import ConflictMetrics, analyze_points
from .geometry import LineKey
from .io import Point
from .state import PairPermutations, points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class LazyRepairRound:
    round_index: int
    status: str
    metrics: ConflictMetrics | None
    new_constraints: int
    total_constraints: int
    hamming_distance: int | None
    solve_seconds: float
    state: PairPermutations | None


@dataclass(frozen=True)
class LazyRepairResult:
    status: str
    state: PairPermutations
    metrics: ConflictMetrics
    rounds: int
    constraints: int
    elapsed_seconds: float


def lazy_repair_state(
    state: PairPermutations,
    free_rows: Iterable[int],
    *,
    max_rounds: int,
    time_limit_per_round: float,
    seed: int,
    workers: int,
    minimize_hamming: bool = True,
    callback: Callable[[LazyRepairRound], None] | None = None,
) -> LazyRepairResult:
    """Iteratively separate conflicts inside a row-LNS neighborhood."""
    state.validate()
    n = state.n
    free = tuple(sorted(set(int(row) for row in free_rows)))
    if not free or any(row < 0 or row >= n for row in free):
        raise ValueError("free_rows must be a nonempty in-grid set")
    free_set = set(free)
    original = set(state.points())
    fixed = sorted(point for point in original if point[0] not in free_set)
    model = cp_model.CpModel()
    variables = {
        (row, column): model.new_bool_var(f"x_{row}_{column}")
        for row in free
        for column in range(n)
    }
    for row in free:
        model.add(sum(variables[row, column] for column in range(n)) == 2)
    fixed_column_counts = [0] * n
    for _, column in fixed:
        fixed_column_counts[column] += 1
    for column in range(n):
        needed = 2 - fixed_column_counts[column]
        if needed < 0:
            raise ValueError("fixed rows overfill a column")
        model.add(sum(variables[row, column] for row in free) == needed)

    for point, variable in variables.items():
        model.add_hint(variable, int(point in original))
    if minimize_hamming:
        model.minimize(
            sum(1 - variable if point in original else variable for point, variable in variables.items())
        )

    seen: set[LineKey] = set()

    def add_line(key: LineKey) -> bool:
        if key in seen:
            return False
        coefficient_a, coefficient_b, coefficient_c = key
        fixed_count = sum(
            coefficient_a * row + coefficient_b * column + coefficient_c == 0
            for row, column in fixed
        )
        on_line = [
            variable
            for (row, column), variable in variables.items()
            if coefficient_a * row + coefficient_b * column + coefficient_c == 0
        ]
        bound = 2 - fixed_count
        if bound < 0:
            model.add_bool_or([])
        elif len(on_line) > bound:
            model.add(sum(on_line) <= bound)
        seen.add(key)
        return True

    best_state = state
    best_metrics = analyze_points(state.points())
    initial_new = sum(add_line(line.key) for line in best_metrics.lines)
    if callback:
        callback(LazyRepairRound(0, "INITIAL", best_metrics, initial_new, len(seen), 0, 0.0, state))
    started = time.perf_counter()

    for round_index in range(1, max_rounds + 1):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_per_round)
        solver.parameters.random_seed = int(seed + round_index - 1)
        solver.parameters.num_search_workers = int(workers)
        solve_started = time.perf_counter()
        status_code = solver.solve(model)
        solve_seconds = time.perf_counter() - solve_started
        status = solver.status_name(status_code)
        if status_code not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            if callback:
                callback(LazyRepairRound(round_index, status, None, 0, len(seen), None, solve_seconds, None))
            return LazyRepairResult(
                status,
                best_state,
                best_metrics,
                round_index,
                len(seen),
                time.perf_counter() - started,
            )

        points = fixed + [point for point, variable in variables.items() if solver.value(variable)]
        candidate = points_to_permutations(points, n)
        metrics = analyze_points(candidate.points())
        if metrics.lexicographic < best_metrics.lexicographic:
            best_state = candidate
            best_metrics = metrics
        hamming = len(original.symmetric_difference(set(points)))
        if metrics.triples == 0:
            ok_a, detail_a = verify_a(points, n)
            ok_b, detail_b = verify_b(points, n)
            if not (ok_a and ok_b):
                raise AssertionError(f"lazy repair witness failed verification: A={detail_a}; B={detail_b}")
            event = LazyRepairRound(round_index, "VALID", metrics, 0, len(seen), hamming, solve_seconds, candidate)
            if callback:
                callback(event)
            return LazyRepairResult(
                "VALID", candidate, metrics, round_index, len(seen), time.perf_counter() - started
            )

        new_constraints = sum(add_line(line.key) for line in metrics.lines)
        event = LazyRepairRound(
            round_index,
            status,
            metrics,
            new_constraints,
            len(seen),
            hamming,
            solve_seconds,
            candidate,
        )
        if callback:
            callback(event)
        if not new_constraints:
            raise AssertionError("conflicts remain but no new lazy repair cut was generated")

    return LazyRepairResult(
        "ROUND_LIMIT",
        best_state,
        best_metrics,
        max_rounds,
        len(seen),
        time.perf_counter() - started,
    )
