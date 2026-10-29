from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from pysat.solvers import Solver

from .sat import _add_atmost
from .sat_assumptions import build_full_cnf
from .state import PairPermutations, points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class DistanceSatResult:
    status: str
    state: PairPermutations | None
    max_removed_points: int
    variables: int
    clauses: int
    line_constraints: int
    model_seconds: float
    solve_seconds: float


def solve_distance_ball(
    state: PairPermutations,
    *,
    max_removed_points: int,
    time_limit: float,
    solver_name: str = "glucose42",
    encoding: str = "seqcounter",
) -> DistanceSatResult:
    """Search the exact global Hamming ball around a 2-factor state.

    Row/column equalities imply that removing ``k`` selected points also adds
    ``k`` unselected points, so the symmetric-difference radius is ``2k``.
    """
    if solver_name.lower().startswith("kissat"):
        raise ValueError(
            "Kissat is unavailable for bounded distance runs: its PySAT wrapper "
            "does not provide the phase/interrupt API required for a reliable timeout"
        )
    state.validate()
    if not 0 <= max_removed_points <= 2 * state.n:
        raise ValueError("max_removed_points outside [0,2n]")
    started_model = time.perf_counter()
    cnf, pool, literal_for, line_constraints, _base_model_seconds = build_full_cnf(
        state.n, encoding=encoding
    )
    selected = set(state.points())
    removal_literals = [-literal_for[point] for point in state.points()]
    _add_atmost(cnf, removal_literals, max_removed_points, pool, encoding="seqcounter")
    model_seconds = time.perf_counter() - started_model
    phases = [literal if point in selected else -literal for point, literal in literal_for.items()]

    found = None
    started_solve = time.perf_counter()
    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as solver:
        try:
            solver.set_phases(phases)
        except NotImplementedError:
            # Kissat's PySAT wrapper does not expose phase selection.  The
            # exact formula is unchanged, so it remains a valid independent
            # portfolio engine without the near-state polarity hint.
            pass
        timer = threading.Timer(float(time_limit), solver.interrupt)
        timer.start()
        try:
            answer = solver.solve_limited(expect_interrupt=True)
        finally:
            timer.cancel()
            try:
                solver.clear_interrupt()
            except NotImplementedError:
                pass
        solve_seconds = time.perf_counter() - started_solve
        if answer is True:
            positive = {literal for literal in solver.get_model() if literal > 0}
            points = [point for point, literal in literal_for.items() if literal in positive]
            ok_a, detail_a = verify_a(points, state.n)
            ok_b, detail_b = verify_b(points, state.n)
            if not (ok_a and ok_b):
                raise AssertionError(f"distance SAT witness failed verification: A={detail_a}; B={detail_b}")
            found = points_to_permutations(points, state.n)
            status = "SAT"
        elif answer is False:
            status = "UNSAT"
        else:
            status = "UNKNOWN"
    return DistanceSatResult(
        status,
        found,
        max_removed_points,
        pool.top,
        len(cnf.clauses),
        line_constraints,
        model_seconds,
        solve_seconds,
    )
