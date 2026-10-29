from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .accelerated import evaluate_two_switches


CUDA_SOURCE = r'''
__device__ __forceinline__ int column_at(const int* p, const int* q, int index) {
    const int row = index >> 1;
    return (index & 1) ? q[row] : p[row];
}

extern "C" __global__
void two_switch_scores(
    const int* p,
    const int* q,
    const int* badness,
    const int n,
    const int u,
    const int current_score,
    int* scores
) {
    const int v = threadIdx.x + blockIdx.x * blockDim.x;
    const int count = 2 * n;
    const int sentinel = 1 << 30;
    if (v >= count) return;
    scores[v] = sentinel;
    const int row_u = u >> 1;
    const int row_v = v >> 1;
    if (v == u || row_v == row_u) return;
    const int other_u = 2 * row_u + (1 - (u & 1));
    const int other_v = 2 * row_v + (1 - (v & 1));
    const int col_u = column_at(p, q, u);
    const int col_v = column_at(p, q, v);
    if (col_v == column_at(p, q, other_u) || col_u == column_at(p, q, other_v)) return;

    int joint_old = 0;
    for (int other = 0; other < count; ++other) {
        if (other == u || other == v) continue;
        const int row_o = other >> 1;
        const int col_o = column_at(p, q, other);
        joint_old += ((row_v - row_u) * (col_o - col_u) == (col_v - col_u) * (row_o - row_u));
    }
    const int old_affected = badness[u] + badness[v] - joint_old;
    int new_affected = 0;

    for (int i = 0; i < count - 1; ++i) {
        if (i == u) continue;
        const int row_i = i >> 1;
        const int col_i = (i == v) ? col_u : column_at(p, q, i);
        for (int j = i + 1; j < count; ++j) {
            if (j == u) continue;
            const int row_j = j >> 1;
            const int col_j = (j == v) ? col_u : column_at(p, q, j);
            new_affected += ((row_i - row_u) * (col_j - col_v) == (col_i - col_v) * (row_j - row_u));
        }
    }
    for (int i = 0; i < count - 1; ++i) {
        if (i == u || i == v) continue;
        const int row_i = i >> 1;
        const int col_i = column_at(p, q, i);
        for (int j = i + 1; j < count; ++j) {
            if (j == u || j == v) continue;
            const int row_j = j >> 1;
            const int col_j = column_at(p, q, j);
            new_affected += ((row_i - row_v) * (col_j - col_u) == (col_i - col_u) * (row_j - row_v));
        }
    }
    scores[v] = current_score - old_affected + new_affected;
}
'''


@lru_cache(maxsize=1)
def _cupy_and_kernel():
    import cupy as cp

    return cp, cp.RawKernel(CUDA_SOURCE, "two_switch_scores", options=("-std=c++11",))


def score_two_switches_gpu(
    p: np.ndarray,
    q: np.ndarray,
    selected_index: int,
    current_score: int,
    badness: np.ndarray,
) -> np.ndarray:
    p_host = np.ascontiguousarray(p, dtype=np.int32)
    q_host = np.ascontiguousarray(q, dtype=np.int32)
    bad_host = np.ascontiguousarray(badness, dtype=np.int32)
    if p_host.ndim != 1 or q_host.shape != p_host.shape or bad_host.shape != (2 * len(p_host),):
        raise ValueError("invalid P/Q/badness shapes")
    cp, kernel = _cupy_and_kernel()
    device_p = cp.asarray(p_host)
    device_q = cp.asarray(q_host)
    device_badness = cp.asarray(bad_host)
    output = cp.empty(2 * len(p_host), dtype=cp.int32)
    kernel(
        (1,),
        (256,),
        (
            device_p,
            device_q,
            device_badness,
            np.int32(len(p_host)),
            np.int32(selected_index),
            np.int32(current_score),
            output,
        ),
    )
    return cp.asnumpy(output)


@dataclass(frozen=True)
class DeltaBenchmark:
    cpu_seconds: float
    gpu_seconds: float
    repeats: int
    exact_match: bool

    @property
    def speedup(self) -> float:
        return self.cpu_seconds / max(self.gpu_seconds, 1e-12)


def benchmark_delta(
    p: np.ndarray,
    q: np.ndarray,
    selected_index: int,
    current_score: int,
    badness: np.ndarray,
    *,
    repeats: int,
) -> DeltaBenchmark:
    cpu_expected = evaluate_two_switches(p, q, selected_index, current_score, badness)
    gpu_actual = score_two_switches_gpu(p, q, selected_index, current_score, badness)
    sentinel_cpu = cpu_expected >= (1 << 59)
    sentinel_gpu = gpu_actual >= (1 << 29)
    exact_match = bool(
        np.array_equal(sentinel_cpu, sentinel_gpu)
        and np.array_equal(cpu_expected[~sentinel_cpu].astype(np.int32), gpu_actual[~sentinel_gpu])
    )

    started = time.perf_counter()
    for _ in range(repeats):
        evaluate_two_switches(p, q, selected_index, current_score, badness)
    cpu_seconds = (time.perf_counter() - started) / repeats
    started = time.perf_counter()
    for _ in range(repeats):
        score_two_switches_gpu(p, q, selected_index, current_score, badness)
    gpu_seconds = (time.perf_counter() - started) / repeats
    return DeltaBenchmark(cpu_seconds, gpu_seconds, repeats, exact_match)
