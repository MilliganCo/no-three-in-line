"""Exhaustive invertible affine coordinate maps modulo n, exactly rescored."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .accelerated import score_and_badness
from .gpu import _cupy_and_kernel
from .io import load_candidate, write_candidate
from .state import points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


def affine_points(points,n,a,b,c,d):
    if math.gcd(a,n)!=1 or math.gcd(c,n)!=1:
        raise ValueError("both multipliers must be invertible modulo n")
    return sorted(((a*x+b)%n,(c*y+d)%n) for x,y in points)


def balanced_primitive_forms(points, n):
    """Find primitive linear forms preserving degree two modulo n.

    Any row (a,b) of an invertible 2x2 matrix modulo n is primitive:
    gcd(a,b,n)=1. Translation only permutes the projection histogram.
    If every balanced form is axis-aligned, every admissible invertible
    affine map is diagonal or anti-diagonal, hence covered by separable
    maps plus a score-preserving coordinate transpose.
    """
    points_to_permutations(points,n)
    coordinates = np.asarray(points,dtype=np.int64)
    forms = []
    checked = 0
    for a in range(n):
        for b in range(n):
            if math.gcd(math.gcd(a,b),n) != 1:
                continue
            checked += 1
            projection = (a*coordinates[:,0]+b*coordinates[:,1])%n
            if np.all(np.bincount(projection,minlength=n) == 2):
                forms.append((a,b))
    return checked, forms


def search(path,output,keep=64,batch_size=32768,translations_only=False):
    n,points,_=load_candidate(path)
    state=points_to_permutations(points,n)
    units=[1] if translations_only else [a for a in range(1,n) if math.gcd(a,n)==1]
    inverse=[pow(a,-1,n) for a in units]
    count=len(units)**2*n*n
    cp,kernel=_cupy_and_kernel()
    p=cp.asarray(state.P,dtype=cp.int32)
    q=cp.asarray(state.Q,dtype=cp.int32)
    inv=cp.asarray(inverse,dtype=cp.int32)
    units_device=cp.asarray(units,dtype=cp.int32)
    rows=cp.arange(n,dtype=cp.int32)[None,:]
    retained=[]
    histogram=np.zeros(math.comb(2*n,3)+1,dtype=np.int64)
    start=time.perf_counter()
    for first in range(0,count,batch_size):
        ids=cp.arange(first,min(first+batch_size,count),dtype=cp.int32)
        d=ids%n
        b=(ids//n)%n
        cindex=(ids//(n*n))%len(units)
        aindex=ids//(n*n*len(units))
        source_rows=((rows-b[:,None])*inv[aindex,None])%n
        pp=cp.ascontiguousarray((p[source_rows]*units_device[cindex,None]+d[:,None])%n)
        qq=cp.ascontiguousarray((q[source_rows]*units_device[cindex,None]+d[:,None])%n)
        scores=cp.empty(len(ids),dtype=cp.int32)
        kernel((len(ids),),(128,),(pp,qq,np.int32(n),scores))
        scores_host=cp.asnumpy(scores)
        histogram+=np.bincount(scores_host,minlength=len(histogram))
        indices=np.argsort(scores_host,kind="stable")[:keep]
        retained.extend((int(scores_host[i]),first+int(i)) for i in indices)
        retained=sorted(retained)[:keep]
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for index,(score,identifier) in enumerate(retained):
        d=identifier%n
        b=(identifier//n)%n
        c=units[(identifier//(n*n))%len(units)]
        a=units[identifier//(n*n*len(units))]
        transformed=affine_points(points,n,a,b,c,d)
        candidate=points_to_permutations(transformed,n)
        exact,_=score_and_badness(np.asarray(candidate.P,dtype=np.int32),np.asarray(candidate.Q,dtype=np.int32))
        if exact!=score:
            raise AssertionError("affine GPU score mismatch")
        write_candidate(output/f"seed-{index:04d}-E{score}.json",n,transformed,
            conflicts=score,source=str(path),affine={"a":a,"b":b,"c":c,"d":d})
        if score==0:
            assert verify_a(transformed,n)[0] and verify_b(transformed,n)[0]
    summary={"n":n,"source":str(path),"evaluations":count,"best":retained[0][0],
        "elapsed_seconds":time.perf_counter()-start,"translations_only":translations_only,
        "histogram":{str(i):int(v) for i,v in enumerate(histogram) if v}}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="histogram"}),flush=True)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("--output",required=True)
    parser.add_argument("--keep",type=int,default=64)
    parser.add_argument("--translations-only",action="store_true")
    args=parser.parse_args()
    search(args.candidate,args.output,args.keep,translations_only=args.translations_only)
