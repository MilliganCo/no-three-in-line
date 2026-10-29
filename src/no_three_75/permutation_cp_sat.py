from __future__ import annotations

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .state import PairPermutations, points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class PermutationCpSatResult:
    status: str
    state: PairPermutations | None
    variables: int
    collinearity_constraints: int
    model_seconds: float
    solve_seconds: float
    conflicts: int
    branches: int


def solve_permutation_model(
    n: int,
    *,
    time_limit: float,
    seed: int,
    workers: int,
    hint: PairPermutations | None = None,
    log_search: bool = False,
) -> PermutationCpSatResult:
    """Solve the complete problem using two integer permutations.

    For rows ``a < b < c`` and selected columns ``u, v, w``, the determinant
    is zero exactly when

        (c-b) u - (c-a) v + (b-a) w == 0.

    Enumerating the eight choices of P/Q for every row triple is therefore a
    complete exact encoding.  Same-row triples cannot occur and the two
    all-different constraints, together with P[r] != Q[r], give exactly two
    points in every column.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if hint is not None:
        hint.validate()
        if hint.n != n:
            raise ValueError("hint size does not match n")

    model_started = time.perf_counter()
    model = cp_model.CpModel()
    p = [model.new_int_var(0, n - 1, f"p_{row}") for row in range(n)]
    q = [model.new_int_var(0, n - 1, f"q_{row}") for row in range(n)]
    model.add_all_different(p)
    model.add_all_different(q)
    for row in range(n):
        model.add(p[row] != q[row])

    # P and Q are interchangeable.  This removes the global colour swap while
    # leaving every uncoloured 2-factor representable.
    model.add(p[0] < q[0])

    layers = (p, q)
    collinearity_constraints = 0
    for row_a in range(n - 2):
        for row_b in range(row_a + 1, n - 1):
            for row_c in range(row_b + 1, n):
                coefficient_a = row_c - row_b
                coefficient_b = -(row_c - row_a)
                coefficient_c = row_b - row_a
                for layer_a in range(2):
                    for layer_b in range(2):
                        for layer_c in range(2):
                            model.add(
                                coefficient_a * layers[layer_a][row_a]
                                + coefficient_b * layers[layer_b][row_b]
                                + coefficient_c * layers[layer_c][row_c]
                                != 0
                            )
                            collinearity_constraints += 1

    if hint is not None:
        hinted_p = list(hint.P)
        hinted_q = list(hint.Q)
        if hinted_p[0] > hinted_q[0]:
            hinted_p, hinted_q = hinted_q, hinted_p
        for row in range(n):
            model.add_hint(p[row], hinted_p[row])
            model.add_hint(q[row], hinted_q[row])
    model_seconds = time.perf_counter() - model_started

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log_search)
    solve_started = time.perf_counter()
    status_code = solver.solve(model)
    solve_seconds = time.perf_counter() - solve_started
    state = None
    if status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        raw_p = tuple(int(solver.value(variable)) for variable in p)
        raw_q = tuple(int(solver.value(variable)) for variable in q)
        state = PairPermutations(raw_p, raw_q)
        state.validate()
        ok_a, detail_a = verify_a(state.points(), n)
        ok_b, detail_b = verify_b(state.points(), n)
        if not (ok_a and ok_b):
            raise AssertionError(
                f"permutation CP-SAT witness failed verification: A={detail_a}; B={detail_b}"
            )
        # Normalize the colouring for stable downstream files.
        state = points_to_permutations(state.points(), n)

    return PermutationCpSatResult(
        solver.status_name(status_code),
        state,
        2 * n,
        collinearity_constraints,
        model_seconds,
        solve_seconds,
        int(solver.num_conflicts),
        int(solver.num_branches),
    )
