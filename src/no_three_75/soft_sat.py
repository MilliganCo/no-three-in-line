"""Exact bounded-triple SAT repair using weighted occupancy thresholds."""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from .conflicts import analyze_points
from .cp_sat import _multi_point_lines, choose_free_rows
from .io import load_candidate, write_candidate
from .sat import _add_atmost, _add_equals
from .state import points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


def repair(state, free_rows, target, seconds, *, force_initial=False):
    state.validate()
    free = sorted(set(free_rows))
    if target < 0 or not free or min(free) < 0 or max(free) >= state.n:
        raise ValueError("invalid target or free rows")
    started = time.perf_counter()
    fixed = [p for p in state.points() if p[0] not in free]
    residual = [2-sum(c == column for _, c in fixed) for column in range(state.n)]
    pool, cnf = IDPool(), CNF()
    variables = {(r,c): pool.id((r,c)) for r in free for c in range(state.n) if residual[c]}
    for r in free:
        _add_equals(cnf, [v for (rr,c),v in variables.items() if rr == r], 2, pool)
    for c in range(state.n):
        if residual[c]:
            _add_equals(cnf, [v for (r,cc),v in variables.items() if cc == c], residual[c], pool)
    active = fixed + list(variables)
    lines = _multi_point_lines(active, state.n)
    maximum = 2
    while math.comb(maximum+1,3) <= target:
        maximum += 1
    penalties, baseline = [], 0
    thresholds = 0
    for members in lines.values():
        occupied = sum(i < len(fixed) for i in members)
        choices = [variables[active[i]] for i in members if i >= len(fixed)]
        # Row/column equalities already exclude triples on axis-aligned lines.
        points = [active[i] for i in members]
        if len({r for r,c in points}) == 1 or len({c for r,c in points}) == 1:
            continue
        baseline += math.comb(occupied,3) if occupied >= 3 else 0
        _add_atmost(cnf, choices, maximum-occupied, pool)
        for k in range(max(3,occupied+1), min(maximum,occupied+len(choices))+1):
            indicator = pool.id()
            layer = CNF()
            bound = k-1-occupied
            _add_atmost(layer, choices, bound, pool,
                        encoding="direct" if bound <= 2 else "seqcounter")
            cnf.extend([clause+[indicator] for clause in layer.clauses])
            # C(t,3) = sum_{k=3}^t C(k-1,2). Repeated input literals
            # implement these integer weights in the sequential counter.
            penalties.extend([indicator]*math.comb(k-1,2))
            thresholds += 1
    _add_atmost(cnf, penalties, target-baseline, pool)
    original = set(state.points())
    phases = [v if p in original else -v for p,v in variables.items()]
    if force_initial:
        cnf.extend([[v] for v in phases])
    model_seconds = time.perf_counter()-started
    started = time.perf_counter()
    result = None
    with Solver(name="glucose42", bootstrap_with=cnf.clauses) as solver:
        solver.set_phases(phases)
        timer = threading.Timer(seconds, solver.interrupt)
        timer.start()
        try:
            answer = solver.solve_limited(expect_interrupt=True)
        finally:
            timer.cancel()
            solver.clear_interrupt()
        payload = {"status": "SAT" if answer is True else "UNSAT" if answer is False else "UNKNOWN",
                   "target": target, "free_rows": free, "variables": pool.top,
                   "grid_variables": len(variables), "clauses": len(cnf.clauses),
                   "thresholds": thresholds, "model_seconds": model_seconds,
                   "solve_seconds": time.perf_counter()-started, "statistics": solver.accum_stats()}
        if answer is True:
            positive = set(v for v in solver.get_model() if v > 0)
            points = fixed+[p for p,v in variables.items() if v in positive]
            result = points_to_permutations(points, state.n)
            actual = analyze_points(points).triples
            if actual > target:
                raise AssertionError(f"soft SAT produced {actual}>{target}")
            payload["actual_triples"] = actual
            if actual == 0:
                assert verify_a(points,state.n)[0] and verify_b(points,state.n)[0]
    return result, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("--free-rows", type=int, required=True)
    parser.add_argument("--target", type=int, default=9)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--strategy", choices=("all-conflicts","cover-random"), default="all-conflicts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    n, points, _ = load_candidate(args.candidate)
    state = points_to_permutations(points,n)
    free = choose_free_rows(state,args.free_rows,seed=args.seed,strategy=args.strategy)
    result, payload = repair(state,free,args.target,args.seconds)
    payload.update(source=args.candidate, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"result.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    if result is not None:
        write_candidate(output/f"candidate-E{payload['actual_triples']}.json",n,result.points(),soft_sat=payload)
    print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__":
    main()
