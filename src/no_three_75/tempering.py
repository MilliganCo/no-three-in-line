"""Persistent CUDA swap chains with replica exchange and exact integer deltas."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .accelerated import score_and_badness
from .gpu_device_search import MUTATION_SOURCE
from .io import load_candidate, write_candidate
from .state import points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


SOURCE = MUTATION_SOURCE.split('extern "C"')[0] + r'''
__device__ __forceinline__ int aligned(int a, int ca, int b, int cb, int c, int cc) {
    return (b-a)*(cc-ca) == (cb-ca)*(c-a);
}
__device__ __forceinline__ void difference(int d, int a, int b, int c, int* db, int* de) {
    if (d) {
        atomicAdd(de, d);
        atomicAdd(db+a, d); atomicAdd(db+b, d); atomicAdd(db+c, d);
    }
}
extern "C" __global__ void walk(
    int* columns, int* badness, int* scores, unsigned long long* rngs,
    int* best_columns, int* best_scores, const float* temperatures,
    int* accepted, const int* reference, int* removed_points,
    const float* centers, const float* strengths,
    const int n, const int steps, const int bad_bias
) {
    const int replica = blockIdx.x;
    const int m = 2*n;
    const int base = replica*m;
    __shared__ int col[150], bad[150], db[150];
    __shared__ int score, u, v, cu, cv, delta, accept, valid, record;
    __shared__ int removed, distance_delta;
    __shared__ unsigned long long rng;
    for (int i=threadIdx.x;i<m;i+=blockDim.x) {
        col[i]=columns[base+i]; bad[i]=badness[base+i];
    }
    if (threadIdx.x==0) {score=scores[replica];rng=rngs[replica];removed=removed_points[replica];}
    __syncthreads();
    for(int step=0;step<steps;++step) {
        if(threadIdx.x==0) {
            u=next_u64(&rng)%m;
            if(bad_bias>0 && score>0 && (int)(next_u64(&rng)%100)<bad_bias) {
                int target=next_u64(&rng)%(3*score);
                u=0;
                while(u<m-1 && target>=bad[u]) {target-=bad[u];++u;}
            }
            v=next_u64(&rng)%m;
            cu=col[u];cv=col[v];
            valid=score>0 && (u>>1)!=(v>>1) && cu!=cv && col[u^1]!=cv && col[v^1]!=cu;
            const int ur=(u>>1)*2,vr=(v>>1)*2;
            distance_delta=(cu==reference[ur] || cu==reference[ur+1])
                +(cv==reference[vr] || cv==reference[vr+1])
                -(cv==reference[ur] || cv==reference[ur+1])
                -(cu==reference[vr] || cu==reference[vr+1]);
            delta=0;accept=0;record=0;
        }
        for(int i=threadIdx.x;i<m;i+=blockDim.x) db[i]=0;
        __syncthreads();
        if(valid) {
            const int ru=u>>1,rv=v>>1;
            // Triples containing exactly one moved point.
            for(int i=threadIdx.x;i<m-1;i+=blockDim.x) {
                if(i==u || i==v) continue;
                const int ri=i>>1,ci=col[i];
                for(int j=i+1;j<m;++j) {
                    if(j==u || j==v) continue;
                    const int rj=j>>1,cj=col[j];
                    int d=aligned(ru,cv,ri,ci,rj,cj)-aligned(ru,cu,ri,ci,rj,cj);
                    difference(d,u,i,j,db,&delta);
                    d=aligned(rv,cu,ri,ci,rj,cj)-aligned(rv,cv,ri,ci,rj,cj);
                    difference(d,v,i,j,db,&delta);
                }
            }
            // Triples containing both moved points, counted once.
            for(int k=threadIdx.x;k<m;k+=blockDim.x) {
                if(k==u || k==v) continue;
                int d=aligned(ru,cv,rv,cu,k>>1,col[k])-aligned(ru,cu,rv,cv,k>>1,col[k]);
                difference(d,u,v,k,db,&delta);
            }
        }
        __syncthreads();
        if(threadIdx.x==0 && valid) {
            const double uniform=((next_u64(&rng)>>11)+0.5)/9007199254740992.0;
            const double offset=removed-centers[replica];
            const double energy_delta=delta+strengths[replica]*(2*offset*distance_delta+distance_delta*distance_delta);
            accept=energy_delta<=0 || log(uniform)<-energy_delta/temperatures[replica];
            if(accept) {
                col[u]=cv;col[v]=cu;score+=delta;removed+=distance_delta;
                atomicAdd(accepted+replica,1);
                if(score<best_scores[replica]) {
                    best_scores[replica]=score;record=1;
                }
            }
        }
        __syncthreads();
        for(int i=threadIdx.x;i<m;i+=blockDim.x) {
            if(accept) bad[i]+=db[i];
            if(record) best_columns[base+i]=col[i];
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<m;i+=blockDim.x) {
        columns[base+i]=col[i];badness[base+i]=bad[i];
    }
    if(threadIdx.x==0) {scores[replica]=score;rngs[replica]=rng;removed_points[replica]=removed;}
}
'''


class TemperingChains:
    def __init__(self, columns: np.ndarray, temperatures: np.ndarray, seed: int):
        import cupy as cp

        columns = np.ascontiguousarray(columns, dtype=np.int32)
        if columns.ndim != 2 or columns.shape[1] % 2 or not 4 <= columns.shape[1] <= 150:
            raise ValueError("columns must have shape (replicas,2n), 2<=n<=75")
        self.n = columns.shape[1] // 2
        temperatures = np.asarray(temperatures, dtype=np.float32)
        if temperatures.shape != (len(columns),) or np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0):
            raise ValueError("one positive finite temperature is required per replica")
        values, badness = [], []
        for row in columns:
            self.to_state(row)
            score, bad = score_and_badness(row[0::2].copy(), row[1::2].copy())
            values.append(score)
            badness.append(bad)
        self.cp = cp
        self.kernel = cp.RawKernel(SOURCE, "walk", options=("-std=c++11",))
        self.columns = cp.asarray(columns)
        self.badness = cp.asarray(np.asarray(badness, dtype=np.int32))
        self.scores = cp.asarray(values, dtype=cp.int32)
        self.best_columns = self.columns.copy()
        self.best_scores = self.scores.copy()
        self.temperatures_host = temperatures
        self.temperatures = cp.asarray(temperatures)
        self.host_rng = np.random.default_rng(seed)
        self.rngs = cp.asarray(self.host_rng.integers(1, 2**64-1, size=len(columns), dtype=np.uint64))
        self.accepted = cp.zeros(len(columns), dtype=cp.int32)
        self.reference=cp.asarray(columns[0])
        reference=set((i//2,int(c)) for i,c in enumerate(columns[0]))
        self.removed_points=cp.asarray([len(reference-set((i//2,int(c)) for i,c in enumerate(row))) for row in columns],dtype=cp.int32)
        self.centers=cp.zeros(len(columns),dtype=cp.float32)
        self.strengths=cp.zeros(len(columns),dtype=cp.float32)
        self.round = 0

    def to_state(self, row: np.ndarray):
        return points_to_permutations([(i//2,int(c)) for i,c in enumerate(row)], self.n)

    def advance(self, steps: int, bad_bias: int = 0) -> None:
        if steps < 1 or not 0 <= bad_bias <= 100:
            raise ValueError("steps must be positive and bad_bias in [0,100]")
        self.kernel((len(self.scores),), (128,), (
            self.columns,self.badness,self.scores,self.rngs,
            self.best_columns,self.best_scores,self.temperatures,self.accepted,
            self.reference,self.removed_points,self.centers,self.strengths,
            np.int32(self.n),np.int32(steps),np.int32(bad_bias),
        ))
        self.cp.cuda.Stream.null.synchronize()

    def exchange(self, ladder_size: int) -> int:
        """Exchange neighboring temperatures within each independent ladder."""
        if ladder_size < 2 or len(self.scores) % ladder_size:
            raise ValueError("ladder_size must divide replica count")
        if bool(self.cp.any(self.strengths).item()):
            raise ValueError("temperature-only exchange is disabled with umbrella potentials")
        scores = self.cp.asnumpy(self.scores)
        permutation = np.arange(len(scores))
        exchanged = 0
        for start in range(0,len(scores),ladder_size):
            for a in range(start+self.round%2,start+ladder_size-1,2):
                b = a+1
                log_ratio = (1/self.temperatures_host[a]-1/self.temperatures_host[b])*(int(scores[a])-int(scores[b]))
                if np.log(self.host_rng.random()) < min(0,log_ratio):
                    permutation[a],permutation[b] = b,a
                    exchanged += 1
        indices = self.cp.asarray(permutation)
        self.columns = self.columns[indices]
        self.badness = self.badness[indices]
        self.scores = self.scores[indices]
        self.removed_points = self.removed_points[indices]
        self.round += 1
        return exchanged

    def validate(self, indices=None) -> None:
        if indices is None:
            indices = range(len(self.scores))
        for i in indices:
            row = self.cp.asnumpy(self.columns[i])
            self.to_state(row)
            score,bad = score_and_badness(row[0::2].copy(),row[1::2].copy())
            if score != int(self.scores[i].item()) or not np.array_equal(bad,self.cp.asnumpy(self.badness[i])):
                raise AssertionError(f"replica {i}: incremental score/badness differs from full determinant check")
            reference=set((j//2,int(c)) for j,c in enumerate(self.cp.asnumpy(self.reference)))
            if len(reference-set((j//2,int(c)) for j,c in enumerate(row)))!=int(self.removed_points[i].item()):
                raise AssertionError("incremental distance mismatch")

    def save(self, path: Path) -> None:
        arrays = {name:self.cp.asnumpy(getattr(self,name)) for name in (
            "columns","badness","scores","rngs","best_columns","best_scores","accepted",
            "reference","removed_points","centers","strengths")}
        np.savez_compressed(path,**arrays,temperatures=self.temperatures_host,
            round=np.asarray(self.round),host_rng=np.asarray(json.dumps(self.host_rng.bit_generator.state)))

    @classmethod
    def restore(cls, path: Path):
        with np.load(path,allow_pickle=False) as saved:
            obj = cls(saved["columns"],saved["temperatures"],0)
            for name in ("columns","badness","scores","rngs","best_columns","best_scores","accepted"):
                setattr(obj,name,obj.cp.asarray(saved[name]))
            for name in ("reference","removed_points","centers","strengths"):
                if name in saved:
                    setattr(obj,name,obj.cp.asarray(saved[name]))
            obj.round = int(saved["round"])
            obj.host_rng.bit_generator.state = json.loads(str(saved["host_rng"]))
        obj.validate()
        return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial",nargs="+",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--replicas",type=int,default=128)
    parser.add_argument("--ladder-size",type=int,default=16)
    parser.add_argument("--tmin",type=float,default=0.2)
    parser.add_argument("--tmax",type=float,default=4)
    parser.add_argument("--rounds",type=int,default=200)
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--bad-bias",type=int,default=0)
    parser.add_argument("--resume")
    parser.add_argument("--umbrella-max",type=float,default=0)
    parser.add_argument("--umbrella-strength",type=float,default=0.2)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    states = []
    n = None
    for path in args.initial:
        size,points,_ = load_candidate(path)
        if n is not None and n!=size:
            raise ValueError("all initial sizes must agree")
        n=size
        state=points_to_permutations(points,n)
        states.append(np.column_stack((state.P,state.Q)).flatten())
    if args.replicas % args.ladder_size or args.ladder_size < 2:
        raise ValueError("replicas must be a multiple of ladder-size >=2")
    if args.resume:
        chains=TemperingChains.restore(Path(args.resume))
        if len(chains.scores)!=args.replicas or chains.n!=n:
            raise ValueError("checkpoint dimensions differ from arguments")
    else:
        columns=np.asarray([states[(i//args.ladder_size)%len(states)] for i in range(args.replicas)],dtype=np.int32)
        temperatures=np.tile(np.geomspace(args.tmin,args.tmax,args.ladder_size),args.replicas//args.ladder_size)
        chains=TemperingChains(columns,temperatures,args.seed)
    if args.umbrella_max:
        chains.centers=chains.cp.asarray(np.repeat(np.linspace(0,args.umbrella_max,args.replicas//args.ladder_size),args.ladder_size),dtype=chains.cp.float32)
        chains.strengths.fill(args.umbrella_strength)
    reference=set((i//2,int(c)) for i,c in enumerate(states[0]))
    start=time.perf_counter()
    best=10**9
    proposals=0
    archive={}
    parameters=vars(args)
    (output/"parameters.json").write_text(json.dumps(parameters,indent=2)+"\n",encoding="utf-8")
    for epoch in range(1,args.rounds+1):
        chains.advance(args.steps,args.bad_bias)
        proposals += args.steps*args.replicas
        exchanges=chains.exchange(args.ladder_size) if not args.umbrella_max else 0
        best_scores=chains.cp.asnumpy(chains.best_scores)
        index=int(np.argmin(best_scores))
        if int(best_scores[index])<best:
            best=int(best_scores[index])
            state=chains.to_state(chains.cp.asnumpy(chains.best_columns[index]))
            exact,_=score_and_badness(np.asarray(state.P,dtype=np.int32),np.asarray(state.Q,dtype=np.int32))
            if exact!=best:
                raise AssertionError("record differs from independent full score")
            write_candidate(output/f"best-E{best}.json",n,state.points(),conflicts=best,
                seed=args.seed,epoch=epoch,proposals=proposals,method="persistent CUDA replica exchange")
            print(f"epoch={epoch} best={best} proposals={proposals} seconds={time.perf_counter()-start:.2f}",flush=True)
            if best==0:
                assert verify_a(state.points(),n)[0] and verify_b(state.points(),n)[0]
                break
        if epoch%10==0 or epoch==args.rounds:
            chains.validate([epoch%args.replicas])
            columns=chains.cp.asnumpy(chains.columns)
            scores=chains.cp.asnumpy(chains.scores)
            distances=[len(reference.symmetric_difference((i//2,int(c)) for i,c in enumerate(row))) for row in columns]
            if args.umbrella_max:
                for i,(score,distance) in enumerate(zip(scores,distances)):
                    bucket=distance//8
                    if int(score)<archive.get(bucket,10**9):
                        archive[bucket]=int(score)
                        candidate=chains.to_state(columns[i])
                        write_candidate(output/f"frontier-bin{bucket:02d}.json",n,candidate.points(),
                            conflicts=int(score),symmetric_difference=distance,epoch=epoch,seed=args.seed)
            payload={"epoch":epoch,"proposals":proposals,"elapsed_seconds":time.perf_counter()-start,
                "best":best,"scores":scores.tolist(),"distances":distances,"exchanges":exchanges,
                "accepted":int(chains.cp.asnumpy(chains.accepted).astype(np.int64).sum())}
            with (output/"history.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(payload)+"\n")
        if epoch%50==0:
            chains.save(output/"checkpoint.npz")
    chains.validate()
    chains.save(output/"checkpoint.npz")
    result={"best":best,"proposals":proposals,"elapsed_seconds":time.perf_counter()-start,
        "accepted":int(chains.cp.asnumpy(chains.accepted).astype(np.int64).sum()),"distance_frontier":archive}
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
