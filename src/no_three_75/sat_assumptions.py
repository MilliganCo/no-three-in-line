from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from .cp_sat import _multi_point_lines, choose_free_rows
from .sat import _add_atmost, _add_equals
from .state import PairPermutations, points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class AssumptionTrial:
    free_row_count: int
    seed: int
    free_rows: tuple[int, ...]
    status: str
    assumptions: int
    solve_seconds: float


@dataclass(frozen=True)
class AssumptionPortfolioResult:
    state: PairPermutations | None
    trials: tuple[AssumptionTrial, ...]
    variables: int
    clauses: int
    line_constraints: int
    model_seconds: float
    solver_setup_seconds: float


def build_full_cnf(n: int, *, encoding: str = "seqcounter") -> tuple[CNF, IDPool, dict[tuple[int, int], int], int, float]:
    started = time.perf_counter()
    pool = IDPool(start_from=1)
    literal_for = {
        (row, column): pool.id((row, column))
        for row in range(n)
        for column in range(n)
    }
    cnf = CNF()
    for row in range(n):
        _add_equals(cnf, [literal_for[row, column] for column in range(n)], 2, pool)
    for column in range(n):
        _add_equals(cnf, [literal_for[row, column] for row in range(n)], 2, pool)
    grid = [(row, column) for row in range(n) for column in range(n)]
    line_constraints = 0
    for members in _multi_point_lines(grid, n).values():
        _add_atmost(
            cnf,
            [literal_for[grid[index]] for index in members],
            2,
            pool,
            encoding=encoding,
        )
        line_constraints += 1
    return cnf, pool, literal_for, line_constraints, time.perf_counter() - started


def run_assumption_portfolio(
    state: PairPermutations,
    *,
    free_row_counts: Iterable[int],
    seeds: Iterable[int],
    time_limit: float,
    solver_name: str = "glucose42",
    encoding: str = "seqcounter",
    row_strategy: str = "all-conflicts",
    callback: Callable[[AssumptionTrial], None] | None = None,
) -> AssumptionPortfolioResult:
    state.validate()
    n = state.n
    cnf, pool, literal_for, line_constraints, model_seconds = build_full_cnf(n, encoding=encoding)
    selected = set(state.points())
    setup_started = time.perf_counter()
    solver = Solver(name=solver_name, bootstrap_with=cnf.clauses)
    solver_setup_seconds = time.perf_counter() - setup_started
    phases = [literal if point in selected else -literal for point, literal in literal_for.items()]
    solver.set_phases(phases)
    trials: list[AssumptionTrial] = []
    witness = None
    try:
        for free_row_count in free_row_counts:
            for seed in seeds:
                free_rows = tuple(
                    choose_free_rows(state, free_row_count, seed=seed, strategy=row_strategy)
                )
                free_set = set(free_rows)
                # Row equality plus these two positive literals fixes every
                # other cell in a frozen row to false by propagation.
                assumptions = [
                    literal_for[point]
                    for point in state.points()
                    if point[0] not in free_set
                ]
                timer = threading.Timer(float(time_limit), solver.interrupt)
                # PySAT's wall-clock interrupt is cooperative.  On a hard
                # incremental Glucose call it can be observed hundreds of
                # seconds late, so also impose a native conflict budget.  The
                # timer remains the primary bound; this is a deterministic
                # fail-safe rather than a proof shortcut.
                solver.conf_budget(max(1_000, int(float(time_limit) * 5_000)))
                timer.start()
                solve_started = time.perf_counter()
                try:
                    answer = solver.solve_limited(assumptions=assumptions, expect_interrupt=True)
                finally:
                    timer.cancel()
                    try:
                        solver.clear_interrupt()
                    except NotImplementedError:
                        pass
                solve_seconds = time.perf_counter() - solve_started
                status = "SAT" if answer is True else "UNSAT" if answer is False else "UNKNOWN"
                trial = AssumptionTrial(
                    free_row_count,
                    int(seed),
                    free_rows,
                    status,
                    len(assumptions),
                    solve_seconds,
                )
                trials.append(trial)
                if callback:
                    callback(trial)
                if answer is True:
                    positive = {literal for literal in solver.get_model() if literal > 0}
                    points = [point for point, literal in literal_for.items() if literal in positive]
                    ok_a, detail_a = verify_a(points, n)
                    ok_b, detail_b = verify_b(points, n)
                    if not (ok_a and ok_b):
                        raise AssertionError(f"assumption SAT witness failed verification: A={detail_a}; B={detail_b}")
                    witness = points_to_permutations(points, n)
                    break
                if answer is None:
                    # Do not reuse a solver after interruption.  Some engines
                    # accept clear_interrupt() but subsequently respond very
                    # slowly or ignore a fresh timer.  Rebuilding preserves the
                    # exact CNF and reliable bounded-run semantics, at the cost
                    # of discarding learned clauses only after UNKNOWN.
                    solver.delete()
                    solver = Solver(name=solver_name, bootstrap_with=cnf.clauses)
                    solver.set_phases(phases)
            if witness is not None:
                break
    finally:
        solver.delete()
    return AssumptionPortfolioResult(
        witness,
        tuple(trials),
        pool.top,
        len(cnf.clauses),
        line_constraints,
        model_seconds,
        solver_setup_seconds,
    )
