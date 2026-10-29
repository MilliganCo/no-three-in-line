from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from .cp_sat import _multi_point_lines
from .io import Point
from .sat import _add_atmost, _add_equals
from .symmetry_cp_sat import fundamental_domain, point_orbit, validate_orbit_partition
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class Rct4SatResult:
    status: str
    points: tuple[Point, ...] | None
    orbit_variables: int
    variables: int
    clauses: int
    line_constraints: int
    model_seconds: float
    solve_seconds: float


def _add_weighted_equals_two(
    cnf: CNF,
    incidence: tuple[tuple[int, int], ...],
    literals: list[int],
    pool: IDPool,
    *,
    selector_key: object,
) -> None:
    """Encode sum(weight_i * literal_i) == 2 for weights one or two."""
    unit = [literals[index] for index, weight in incidence if weight == 1]
    heavy = [literals[index] for index, weight in incidence if weight == 2]
    unsupported = [(index, weight) for index, weight in incidence if weight not in (1, 2)]
    if unsupported:
        raise ValueError(f"unsupported row incidence weights: {unsupported}")
    if not heavy:
        _add_equals(cnf, unit, 2, pool)
        return

    _add_atmost(cnf, heavy, 1, pool)
    heavy_selected = pool.id(("heavy-selected", selector_key))
    for literal in heavy:
        cnf.append([-literal, heavy_selected])
    cnf.append([-heavy_selected, *heavy])
    for literal in unit:
        cnf.append([-heavy_selected, -literal])
    _add_atmost(cnf, unit, 2, pool)
    # If no weight-two orbit is selected, exactly two unit orbits are required.
    if len(unit) < 2:
        cnf.append([heavy_selected])
    else:
        at_least_two = CardEnc.atleast(lits=unit, bound=2, vpool=pool, encoding=EncType.seqcounter)
        cnf.extend([[heavy_selected, *clause] for clause in at_least_two.clauses])


def _add_weighted_atmost_two(
    cnf: CNF,
    incidence: tuple[tuple[int, int], ...],
    literals: list[int],
    pool: IDPool,
    *,
    encoding: str,
) -> None:
    """Encode sum(weight_i * literal_i) <= 2 for orbit incidences."""
    unit: list[int] = []
    heavy: list[int] = []
    for index, weight in incidence:
        literal = literals[index]
        if weight >= 3:
            cnf.append([-literal])
        elif weight == 2:
            heavy.append(literal)
        elif weight == 1:
            unit.append(literal)
        else:
            raise ValueError(f"invalid incidence coefficient {weight}")
    _add_atmost(cnf, unit, 2, pool, encoding=encoding)
    _add_atmost(cnf, heavy, 1, pool, encoding=encoding)
    # A weight-two orbit saturates the line and excludes every unit orbit.
    cnf.extend([[-heavy_literal, -unit_literal] for heavy_literal in heavy for unit_literal in unit])


def solve_rct4_sat(
    n: int,
    *,
    time_limit: float,
    solver_name: str = "glucose42",
    encoding: str = "seqcounter",
    hint_points: list[Point] | None = None,
    force_hint: bool = False,
) -> Rct4SatResult:
    """Solve the complete Prellberg rct4/rot4 orbit model as CNF."""
    validate_orbit_partition(n)
    started_model = time.perf_counter()
    reps = fundamental_domain(n)
    orbits = [point_orbit(rep, n) for rep in reps]
    pool = IDPool(start_from=1)
    literals = [pool.id(("orbit", index)) for index in range(len(reps))]
    cnf = CNF()

    for row in range(n):
        incidence = tuple(
            (index, coefficient)
            for index, orbit in enumerate(orbits)
            if (coefficient := sum(point[0] == row for point in orbit))
        )
        _add_weighted_equals_two(cnf, incidence, literals, pool, selector_key=row)

    rep_for_point = {
        point: index
        for index, orbit in enumerate(orbits)
        for point in orbit
    }
    grid = [(row, column) for row in range(n) for column in range(n)]
    seen_incidence: set[tuple[tuple[int, int], ...]] = set()
    for members in _multi_point_lines(grid, n).values():
        counts: dict[int, int] = {}
        for point_index in members:
            rep_index = rep_for_point.get(grid[point_index])
            if rep_index is not None:
                counts[rep_index] = counts.get(rep_index, 0) + 1
        incidence = tuple(sorted(counts.items()))
        if sum(counts.values()) <= 2 or incidence in seen_incidence:
            continue
        seen_incidence.add(incidence)
        _add_weighted_atmost_two(cnf, incidence, literals, pool, encoding=encoding)

    selected_reps: set[int] = set()
    if hint_points is not None:
        selected = set(hint_points)
        expanded: set[Point] = set()
        for index, orbit in enumerate(orbits):
            chosen = sum(point in selected for point in orbit)
            if chosen not in (0, len(orbit)):
                raise ValueError("rct4 SAT hint is not a union of complete model orbits")
            if chosen:
                selected_reps.add(index)
                expanded.update(orbit)
        if expanded != selected:
            raise ValueError("rct4 SAT hint contains points outside the represented orbit domain")
    phases = [literal if index in selected_reps else -literal for index, literal in enumerate(literals)]
    model_seconds = time.perf_counter() - started_model

    points = None
    started_solve = time.perf_counter()
    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as solver:
        if hint_points is not None:
            solver.set_phases(phases)
        timer = threading.Timer(float(time_limit), solver.interrupt)
        timer.start()
        assumptions = None
        if force_hint:
            if hint_points is None:
                raise ValueError("force_hint requires hint_points")
            assumptions = [
                literal if index in selected_reps else -literal
                for index, literal in enumerate(literals)
            ]
        try:
            answer = solver.solve_limited(assumptions=assumptions or [], expect_interrupt=True)
        finally:
            timer.cancel()
            try:
                solver.clear_interrupt()
            except NotImplementedError:
                pass
        solve_seconds = time.perf_counter() - started_solve
        if answer is True:
            positive = {literal for literal in solver.get_model() if literal > 0}
            points = tuple(
                sorted(
                    point
                    for index, orbit in enumerate(orbits)
                    if literals[index] in positive
                    for point in orbit
                )
            )
            ok_a, detail_a = verify_a(list(points), n)
            ok_b, detail_b = verify_b(list(points), n)
            if not (ok_a and ok_b):
                raise AssertionError(f"rct4 SAT witness failed verification: A={detail_a}; B={detail_b}")
            status = "SAT"
        elif answer is False:
            status = "UNSAT"
        else:
            status = "UNKNOWN"

    return Rct4SatResult(
        status,
        points,
        len(reps),
        pool.top,
        len(cnf.clauses),
        len(seen_incidence),
        model_seconds,
        solve_seconds,
    )
