"""Exact bounded-conflict row repair: search for E<=target, including E>0."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from ortools.sat.python import cp_model

from .conflicts import analyze_points
from .cp_sat import _multi_point_lines, choose_free_rows
from .io import load_candidate, write_candidate
from .state import points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


def repair(state, free_rows, target: int, seconds: float, seed: int, workers: int):
    state.validate()
    if target < 0:
        raise ValueError("target must be nonnegative")
    free=sorted(set(free_rows))
    if not free or min(free)<0 or max(free)>=state.n:
        raise ValueError("invalid free rows")
    start=time.perf_counter()
    n=state.n
    original=set(state.points())
    fixed=sorted(p for p in original if p[0] not in free)
    residual=[2-sum(c==column for _,c in fixed) for column in range(n)]
    model=cp_model.CpModel()
    variables={(r,c):model.new_bool_var(f"x{r}_{c}") for r in free for c in range(n) if residual[c]>0}
    for r in free:
        model.add(sum(v for (rr,c),v in variables.items() if rr==r)==2)
    for c in range(n):
        if residual[c]:
            model.add(sum(v for (r,cc),v in variables.items() if cc==c)==residual[c])
    active=fixed+list(variables)
    fixed_count=len(fixed)
    costs=[]
    constant=0
    maximum_occupancy=2
    while math.comb(maximum_occupancy+1,3)<=target:
        maximum_occupancy+=1
    lines=_multi_point_lines(active,n)
    unary_weights={point:0 for point in variables}
    for indices in lines.values():
        occupied=sum(i<fixed_count for i in indices)
        choices=[variables[active[i]] for i in indices if i>=fixed_count]
        if occupied>=2:
            for i in indices:
                if i>=fixed_count:
                    unary_weights[active[i]]+=math.comb(occupied,2)
        if not choices:
            constant+=math.comb(occupied,3) if occupied>=3 else 0
            continue
        total=occupied+sum(choices)
        upper=min(occupied+len(choices),maximum_occupancy)
        model.add(total<=maximum_occupancy)
        if upper<3:
            continue
        cost=model.new_int_var(0,math.comb(upper,3),f"e{len(costs)}")
        # At integer occupancy k, C(k,3) is the maximum of these
        # supporting lines and zero. The summed upper bound forces
        # the true triple total <=target without any approximation.
        for k in range(3,upper+1):
            slope=math.comb(k-1,2)
            model.add(cost>=slope*(total-k)+math.comb(k,3))
        costs.append(cost)
    model.add(sum(costs)+constant<=target)
    # Every newly chosen point already contributes one triple for each
    # pair of fixed points collinear with it. Aggregate this implied bound
    # explicitly instead of leaving it distributed over separate costs.
    model.add(sum(unary_weights[p]*v for p,v in variables.items())+constant<=target)
    forbidden=0
    for point,weight in unary_weights.items():
        if weight+constant>target:
            model.add(variables[point]==0)
            forbidden+=1
    for point,var in variables.items():
        model.add_hint(var,int(point in original))
    build=time.perf_counter()-start
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=seconds
    solver.parameters.random_seed=seed
    solver.parameters.num_search_workers=workers
    started_solve=time.perf_counter()
    status=solver.solve(model)
    payload={"status":solver.status_name(status),"target":target,"free_rows":free,
        "grid_variables":len(variables),"cost_variables":len(costs),"lines":len(lines),
        "model_seconds":build,"solve_seconds":time.perf_counter()-started_solve,
        "branches":solver.num_branches,"conflicts":solver.num_conflicts,"seed":seed,
        "unary_forbidden_cells":forbidden}
    result=None
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        points=fixed+[p for p,v in variables.items() if solver.value(v)]
        result=points_to_permutations(points,n)
        actual=analyze_points(points).triples
        if actual>target:
            raise AssertionError(f"soft model produced {actual}>{target}")
        payload["actual_triples"]=actual
        if actual==0:
            assert verify_a(points,n)[0] and verify_b(points,n)[0]
    return result,payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("--free-rows",type=int,required=True)
    parser.add_argument("--target",type=int,default=9)
    parser.add_argument("--seconds",type=float,default=120)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--strategy",choices=("all-conflicts","cover-random"),default="all-conflicts")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    n,points,_=load_candidate(args.candidate)
    state=points_to_permutations(points,n)
    free=choose_free_rows(state,args.free_rows,seed=args.seed,strategy=args.strategy)
    result,payload=repair(state,free,args.target,args.seconds,args.seed,args.workers)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    payload["source"]=args.candidate
    (output/"result.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    if result is not None:
        write_candidate(output/f"candidate-E{payload['actual_triples']}.json",n,result.points(),soft_repair=payload)
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":
    main()
