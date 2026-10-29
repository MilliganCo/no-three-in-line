from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from .cp_sat import _multi_point_lines
from .state import PairPermutations, points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class SatRepairResult:
    status: str
    state: PairPermutations | None
    free_rows: tuple[int, ...]
    variables: int
    clauses: int
    line_constraints: int
    model_seconds: float
    solve_seconds: float


def _add_equals(cnf: CNF, literals: list[int], bound: int, pool: IDPool) -> None:
    if not (0 <= bound <= len(literals)):
        cnf.append([])
        return
    encoded = CardEnc.equals(lits=literals, bound=bound, vpool=pool, encoding=EncType.seqcounter)
    cnf.extend(encoded.clauses)


def _add_atmost(
    cnf: CNF,
    literals: list[int],
    bound: int,
    pool: IDPool,
    *,
    encoding: str = "seqcounter",
) -> None:
    if bound < 0:
        cnf.append([])
    elif bound == 0:
        cnf.extend([[-literal] for literal in literals])
    elif len(literals) > bound:
        if encoding == "direct" and bound == 1:
            cnf.extend([[-left, -right] for left, right in combinations(literals, 2)])
        elif encoding == "direct" and bound == 2:
            cnf.extend([[-left, -middle, -right] for left, middle, right in combinations(literals, 3)])
        elif bound == 1 and len(literals) <= 4:
            cnf.extend([[-left, -right] for left, right in combinations(literals, 2)])
        elif bound == 2 and len(literals) <= 5:
            cnf.extend([[-left, -middle, -right] for left, middle, right in combinations(literals, 3)])
        else:
            encodings = {
                "seqcounter": EncType.seqcounter,
                "totalizer": EncType.totalizer,
                "cardnetwrk": EncType.cardnetwrk,
            }
            if encoding not in encodings:
                raise ValueError(f"unknown cardinality encoding {encoding}")
            encoded = CardEnc.atmost(lits=literals, bound=bound, vpool=pool, encoding=encodings[encoding])
            cnf.extend(encoded.clauses)


def repair_state_sat(
    state: PairPermutations,
    free_rows: Iterable[int],
    *,
    time_limit: float,
    solver_name: str = "glucose42",
    encoding: str = "seqcounter",
) -> SatRepairResult:
    state.validate()
    n = state.n
    free = tuple(sorted(set(int(row) for row in free_rows)))
    if not free or any(row < 0 or row >= n for row in free):
        raise ValueError("free rows must be a nonempty in-grid set")
    free_set = set(free)
    fixed = [point for point in state.points() if point[0] not in free_set]
    started_model = time.perf_counter()
    pool = IDPool(start_from=1)
    literal_for = {(row, column): pool.id((row, column)) for row in free for column in range(n)}
    cnf = CNF()

    for row in free:
        _add_equals(cnf, [literal_for[row, column] for column in range(n)], 2, pool)
    fixed_column_counts = [0] * n
    for _, column in fixed:
        fixed_column_counts[column] += 1
    for column in range(n):
        _add_equals(
            cnf,
            [literal_for[row, column] for row in free],
            2 - fixed_column_counts[column],
            pool,
        )

    active = fixed + [(row, column) for row in free for column in range(n)]
    fixed_count = len(fixed)
    line_groups = _multi_point_lines(active, n)
    line_constraints = 0
    for members in line_groups.values():
        fixed_on_line = sum(index < fixed_count for index in members)
        variable_points = [active[index] for index in members if index >= fixed_count]
        if fixed_on_line >= 3:
            cnf.append([])
            line_constraints += 1
        elif len(variable_points) > 2 - fixed_on_line:
            _add_atmost(
                cnf,
                [literal_for[point] for point in variable_points],
                2 - fixed_on_line,
                pool,
                encoding=encoding,
            )
            line_constraints += 1
    model_seconds = time.perf_counter() - started_model

    current = set(state.points())
    phases = [literal if point in current else -literal for point, literal in literal_for.items()]
    selected_state = None
    started_solve = time.perf_counter()
    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as solver:
        solver.set_phases(phases)
        timer = threading.Timer(float(time_limit), solver.interrupt)
        timer.start()
        try:
            try:
                answer = solver.solve_limited(expect_interrupt=True)
            except NotImplementedError:
                # Some bundled engines expose neither limited solving nor an
                # interrupt hook. This fallback is useful for short benchmark
                # instances; hard runs should use an interruptible solver.
                timer.cancel()
                answer = solver.solve()
        finally:
            timer.cancel()
            try:
                solver.clear_interrupt()
            except NotImplementedError:
                pass
        solve_seconds = time.perf_counter() - started_solve
        if answer is True:
            status = "SAT"
            positive = {literal for literal in solver.get_model() if literal > 0}
            points = list(fixed)
            points.extend(point for point, literal in literal_for.items() if literal in positive)
            ok_a, detail_a = verify_a(points, n)
            ok_b, detail_b = verify_b(points, n)
            if not (ok_a and ok_b):
                raise AssertionError(f"SAT repair failed verification: A={detail_a}; B={detail_b}")
            selected_state = points_to_permutations(points, n)
        elif answer is False:
            status = "UNSAT"
        else:
            status = "UNKNOWN"
    return SatRepairResult(
        status,
        selected_state,
        free,
        pool.top,
        len(cnf.clauses),
        line_constraints,
        model_seconds,
        solve_seconds,
    )
