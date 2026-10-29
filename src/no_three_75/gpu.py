from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .accelerated import score_and_badness


CUDA_SOURCE = r'''
extern "C" __global__
void score_population(const int* p, const int* q, const int n, int* scores) {
    const int state = blockIdx.x;
    const int count = 2 * n;
    const int base = state * n;
    unsigned int local = 0;
    for (int i = threadIdx.x; i < count - 2; i += blockDim.x) {
        const int ri = i >> 1;
        const int ci = (i & 1) ? q[base + ri] : p[base + ri];
        for (int j = i + 1; j < count - 1; ++j) {
            const int rj = j >> 1;
            const int cj = (j & 1) ? q[base + rj] : p[base + rj];
            const int dr1 = rj - ri;
            const int dc1 = cj - ci;
            for (int k = j + 1; k < count; ++k) {
                const int rk = k >> 1;
                const int ck = (k & 1) ? q[base + rk] : p[base + rk];
                local += (dr1 * (ck - ci) == dc1 * (rk - ri));
            }
        }
    }
    __shared__ unsigned int partial[256];
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[state] = (int)partial[0];
}
'''


@lru_cache(maxsize=1)
def _cupy_and_kernel():
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device is available")
    return cp, cp.RawKernel(CUDA_SOURCE, "score_population", options=("-std=c++11",))


def gpu_device_name() -> str:
    cp, _ = _cupy_and_kernel()
    raw = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _validate_batches(p_batch: np.ndarray, q_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.ascontiguousarray(p_batch, dtype=np.int32)
    q = np.ascontiguousarray(q_batch, dtype=np.int32)
    if p.ndim != 2 or q.shape != p.shape or p.shape[1] < 2:
        raise ValueError("P and Q must have the same two-dimensional shape")
    return p, q


def score_population_gpu(p_batch: np.ndarray, q_batch: np.ndarray) -> np.ndarray:
    """Return exact integer triple counts for a population of P/Q states."""
    p, q = _validate_batches(p_batch, q_batch)
    cp, kernel = _cupy_and_kernel()
    device_p = cp.asarray(p)
    device_q = cp.asarray(q)
    output = cp.empty(p.shape[0], dtype=cp.int32)
    kernel((p.shape[0],), (128,), (device_p, device_q, np.int32(p.shape[1]), output))
    return cp.asnumpy(output)


@dataclass(frozen=True)
class GpuBenchmark:
    device: str
    population: int
    n: int
    cpu_seconds: float
    gpu_end_to_end_seconds: float
    gpu_kernel_seconds: float
    exact_match: bool

    @property
    def end_to_end_speedup(self) -> float:
        return self.cpu_seconds / max(self.gpu_end_to_end_seconds, 1e-12)

    @property
    def kernel_speedup(self) -> float:
        return self.cpu_seconds / max(self.gpu_kernel_seconds, 1e-12)


def benchmark_gpu(
    p_batch: np.ndarray,
    q_batch: np.ndarray,
    *,
    repeats: int = 5,
) -> GpuBenchmark:
    p, q = _validate_batches(p_batch, q_batch)
    score_and_badness(p[0], q[0])
    cpu_started = time.perf_counter()
    cpu_scores = np.asarray([score_and_badness(left, right)[0] for left, right in zip(p, q)], dtype=np.int32)
    cpu_seconds = time.perf_counter() - cpu_started

    score_population_gpu(p[:1], q[:1])
    e2e_started = time.perf_counter()
    gpu_scores = None
    for _ in range(repeats):
        gpu_scores = score_population_gpu(p, q)
    gpu_end_to_end_seconds = (time.perf_counter() - e2e_started) / repeats

    cp, kernel = _cupy_and_kernel()
    device_p = cp.asarray(p)
    device_q = cp.asarray(q)
    output = cp.empty(p.shape[0], dtype=cp.int32)
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        kernel((p.shape[0],), (128,), (device_p, device_q, np.int32(p.shape[1]), output))
    end.record()
    end.synchronize()
    gpu_kernel_seconds = float(cp.cuda.get_elapsed_time(start, end)) / 1000.0 / repeats
    kernel_scores = cp.asnumpy(output)
    exact_match = bool(np.array_equal(cpu_scores, gpu_scores) and np.array_equal(cpu_scores, kernel_scores))
    return GpuBenchmark(
        gpu_device_name(),
        p.shape[0],
        p.shape[1],
        cpu_seconds,
        gpu_end_to_end_seconds,
        gpu_kernel_seconds,
        exact_match,
    )
