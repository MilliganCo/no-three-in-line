from __future__ import annotations

import time
from functools import lru_cache
from random import Random
from typing import Callable

import numpy as np

from .gpu import CUDA_SOURCE as SCORE_SOURCE
from .gpu_search import GpuSearchResult, GpuSearchSnapshot
from .state import PairPermutations, points_to_permutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


MUTATION_SOURCE = r'''
__device__ __forceinline__ unsigned long long next_u64(unsigned long long* state) {
    unsigned long long x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

extern "C" __global__
void mutate_population(
    const int* parent_p,
    const int* parent_q,
    const int* parent_badness,
    const int elite_count,
    const int quality_count,
    const int n,
    const int population,
    const int active_strength,
    const int max_strength,
    const int bad_bias,
    const unsigned long long base_seed,
    const int generation,
    int* child_p,
    int* child_q
) {
    const int child = blockIdx.x * blockDim.x + threadIdx.x;
    if (child >= population) return;
    unsigned long long rng = base_seed
        ^ (0x9e3779b97f4a7c15ULL * (unsigned long long)(child + 1))
        ^ (0xbf58476d1ce4e5b9ULL * (unsigned long long)(generation + 1));
    int parent;
    if (child < elite_count) {
        parent = child;
    } else {
        const unsigned long long choice = next_u64(&rng);
        const int pool = ((choice & 255ULL) < 160ULL) ? quality_count : elite_count;
        parent = (int)(next_u64(&rng) % (unsigned long long)pool);
    }
    const int source = parent * n;
    const int bad_source = parent * 2 * n;
    const int target = child * n;
    for (int row = 0; row < n; ++row) {
        child_p[target + row] = parent_p[source + row];
        child_q[target + row] = parent_q[source + row];
    }
    if (child < elite_count) return;

    int strength;
    if ((next_u64(&rng) & 31ULL) == 0ULL) {
        strength = 1 + (int)(next_u64(&rng) % (unsigned long long)max_strength);
    } else {
        strength = 1 + (int)(next_u64(&rng) % (unsigned long long)active_strength);
    }
    const int edge_count = 2 * n;
    for (int step = 0; step < strength; ++step) {
        for (int attempt = 0; attempt < 32; ++attempt) {
            int u;
            if ((int)(next_u64(&rng) % 100ULL) < bad_bias) {
                int total_badness = 0;
                for (int edge = 0; edge < edge_count; ++edge) {
                    total_badness += parent_badness[bad_source + edge];
                }
                if (total_badness) {
                    int threshold = (int)(next_u64(&rng) % (unsigned long long)total_badness);
                    u = 0;
                    for (; u < edge_count - 1; ++u) {
                        threshold -= parent_badness[bad_source + u];
                        if (threshold < 0) break;
                    }
                } else {
                    u = (int)(next_u64(&rng) % (unsigned long long)edge_count);
                }
            } else {
                u = (int)(next_u64(&rng) % (unsigned long long)edge_count);
            }
            const int v = (int)(next_u64(&rng) % (unsigned long long)edge_count);
            const int row_u = u >> 1;
            const int row_v = v >> 1;
            if (row_u == row_v) continue;
            const int layer_u = u & 1;
            const int layer_v = v & 1;
            int* slot_u = layer_u ? child_q + target + row_u : child_p + target + row_u;
            int* slot_v = layer_v ? child_q + target + row_v : child_p + target + row_v;
            const int column_u = *slot_u;
            const int column_v = *slot_v;
            if (column_u == column_v) continue;
            const int other_u = layer_u ? child_p[target + row_u] : child_q[target + row_u];
            const int other_v = layer_v ? child_p[target + row_v] : child_q[target + row_v];
            if (column_v == other_u || column_u == other_v) continue;
            *slot_u = column_v;
            *slot_v = column_u;
            break;
        }
    }
}
'''


SCORE_BADNESS_SOURCE = r'''
extern "C" __global__
void score_population_badness(
    const int* p,
    const int* q,
    const int n,
    int* scores,
    int* badness
) {
    const int state = blockIdx.x;
    const int count = 2 * n;
    const int base = state * n;
    __shared__ unsigned int local_score;
    __shared__ int local_badness[150];
    if (threadIdx.x == 0) local_score = 0;
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        local_badness[index] = 0;
    }
    __syncthreads();
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
                if (dr1 * (ck - ci) == dc1 * (rk - ri)) {
                    atomicAdd(&local_score, 1U);
                    atomicAdd(&local_badness[i], 1);
                    atomicAdd(&local_badness[j], 1);
                    atomicAdd(&local_badness[k], 1);
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) scores[state] = (int)local_score;
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        badness[state * count + index] = local_badness[index];
    }
}
'''


@lru_cache(maxsize=1)
def _kernels():
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device is available")
    return (
        cp,
        cp.RawKernel(MUTATION_SOURCE, "mutate_population", options=("-std=c++11",)),
        cp.RawKernel(SCORE_BADNESS_SOURCE, "score_population_badness", options=("-std=c++11",)),
        cp.RawKernel(SCORE_SOURCE, "score_population", options=("-std=c++11",)),
    )


def _normalize_raw(p: np.ndarray, q: np.ndarray) -> PairPermutations:
    n = p.shape[0]
    points = [(row, int(p[row])) for row in range(n)] + [(row, int(q[row])) for row in range(n)]
    return points_to_permutations(points, n)


def rank_elites(scores: np.ndarray, badness: np.ndarray) -> np.ndarray:
    """Lexicographic score, max point badness, then squared badness mass."""
    return np.asarray(
        sorted(
            range(len(scores)),
            key=lambda index: (
                int(scores[index]),
                int(badness[index].max()),
                int(np.dot(badness[index].astype(np.int64), badness[index].astype(np.int64))),
            ),
        ),
        dtype=np.int32,
    )


def elite_key(score: int, badness: np.ndarray) -> tuple[int, int, int]:
    values = badness.astype(np.int64)
    return int(score), int(values.max()), int(np.dot(values, values))


def mutate_population_device(
    parent_p: np.ndarray,
    parent_q: np.ndarray,
    *,
    population: int,
    active_strength: int,
    max_strength: int,
    seed: int,
    generation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Test-facing wrapper for the degree-preserving CUDA mutation kernel."""
    if parent_p.ndim != 2 or parent_q.shape != parent_p.shape or parent_p.shape[1] > 75:
        raise ValueError("P/Q batches must match and n must not exceed 75")
    cp, mutation_kernel, badness_kernel, _ = _kernels()
    p = cp.asarray(np.ascontiguousarray(parent_p, dtype=np.int32))
    q = cp.asarray(np.ascontiguousarray(parent_q, dtype=np.int32))
    parent_scores = cp.empty(p.shape[0], dtype=cp.int32)
    parent_badness = cp.empty((p.shape[0], 2 * p.shape[1]), dtype=cp.int32)
    badness_kernel(
        (p.shape[0],),
        (128,),
        (p, q, np.int32(p.shape[1]), parent_scores, parent_badness),
    )
    child_p = cp.empty((population, p.shape[1]), dtype=cp.int32)
    child_q = cp.empty_like(child_p)
    quality_count = max(1, min(p.shape[0], p.shape[0] // 4))
    mutation_kernel(
        ((population + 255) // 256,),
        (256,),
        (
            p,
            q,
            parent_badness,
            np.int32(p.shape[0]),
            np.int32(quality_count),
            np.int32(p.shape[1]),
            np.int32(population),
            np.int32(active_strength),
            np.int32(max_strength),
            np.int32(85),
            np.uint64(seed),
            np.int32(generation),
            child_p,
            child_q,
        ),
    )
    return cp.asnumpy(child_p), cp.asnumpy(child_q)


def gpu_device_beam_search(
    initial: PairPermutations,
    *,
    seed: int,
    generations: int,
    population: int,
    elite_size: int,
    max_strength: int,
    energy_window: int,
    initial_elites: np.ndarray | None = None,
    report_every: int = 5,
    bad_bias: int = 85,
    callback: Callable[[GpuSearchSnapshot], None] | None = None,
) -> GpuSearchResult:
    """Generate mutations, score, and retain the beam without population transfers."""
    initial.validate()
    if initial.n > 75:
        raise ValueError("device badness kernel currently supports n <= 75")
    if not (1 <= elite_size <= population):
        raise ValueError("elite_size must be between one and population")
    if max_strength < 1:
        raise ValueError("max_strength must be positive")
    if not 0 <= bad_bias <= 100:
        raise ValueError("bad_bias must be between zero and 100")
    cp, mutation_kernel, badness_kernel, score_kernel = _kernels()
    rng = Random(seed)
    base = np.column_stack((initial.P, initial.Q)).astype(np.int32)[None, :, :]
    if initial_elites is not None:
        supplied = np.ascontiguousarray(initial_elites, dtype=np.int32)
        if supplied.ndim != 3 or supplied.shape[1:] != (initial.n, 2):
            raise ValueError("initial_elites must have shape (count,n,2)")
        for values in supplied:
            _normalize_raw(values[:, 0], values[:, 1]).validate()
        base = np.concatenate((base, supplied), axis=0)
    base = base[:elite_size]
    elite_p = cp.asarray(np.ascontiguousarray(base[:, :, 0]))
    elite_q = cp.asarray(np.ascontiguousarray(base[:, :, 1]))
    elite_scores_device = cp.empty(len(base), dtype=cp.int32)
    elite_badness = cp.empty((len(base), 2 * initial.n), dtype=cp.int32)
    score_kernel((len(base),), (128,), (elite_p, elite_q, np.int32(initial.n), elite_scores_device))
    badness_scores = cp.empty(len(base), dtype=cp.int32)
    badness_kernel(
        (len(base),),
        (128,),
        (elite_p, elite_q, np.int32(initial.n), badness_scores, elite_badness),
    )
    elite_scores = cp.asnumpy(elite_scores_device)
    elite_badness_host = cp.asnumpy(elite_badness)
    order = rank_elites(elite_scores, elite_badness_host)
    elite_p = elite_p[cp.asarray(order)]
    elite_q = elite_q[cp.asarray(order)]
    elite_badness = elite_badness[cp.asarray(order)]
    elite_scores = elite_scores[order]
    elite_badness_host = elite_badness_host[order]
    best_score = int(elite_scores[0])
    best_key = elite_key(best_score, elite_badness_host[0])
    best_p = cp.asnumpy(elite_p[0])
    best_q = cp.asnumpy(elite_q[0])
    evaluations = len(base)
    last_improvement = 0
    started = time.perf_counter()

    completed = 0
    for generation in range(1, generations + 1):
        completed = generation
        child_p = cp.empty((population, initial.n), dtype=cp.int32)
        child_q = cp.empty_like(child_p)
        scores_device = cp.empty(population, dtype=cp.int32)
        active_strength = min(max_strength, 2 + (generation - last_improvement) // 10)
        quality_count = max(1, min(len(elite_scores), elite_size // 4))
        mutation_kernel(
            ((population + 255) // 256,),
            (256,),
            (
                elite_p,
                elite_q,
                elite_badness,
                np.int32(len(elite_scores)),
                np.int32(quality_count),
                np.int32(initial.n),
                np.int32(population),
                np.int32(active_strength),
                np.int32(max_strength),
                np.int32(bad_bias),
                np.uint64(seed),
                np.int32(generation),
                child_p,
                child_q,
            ),
        )
        score_kernel((population,), (128,), (child_p, child_q, np.int32(initial.n), scores_device))
        scores = cp.asnumpy(scores_device)
        evaluations += population
        order = np.argsort(scores, kind="stable")
        quality_keep = max(1, elite_size // 4)
        selected = list(map(int, order[:quality_keep]))
        selected_set = set(selected)
        diverse = list(map(int, np.flatnonzero(scores <= best_score + energy_window)))
        rng.shuffle(diverse)
        for index in diverse:
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
                if len(selected) == elite_size:
                    break
        if len(selected) < elite_size:
            for raw_index in order:
                index = int(raw_index)
                if index not in selected_set:
                    selected.append(index)
                    selected_set.add(index)
                    if len(selected) == elite_size:
                        break
        selected_array = cp.asarray(np.asarray(selected, dtype=np.int32))
        elite_p = child_p[selected_array]
        elite_q = child_q[selected_array]
        elite_scores = scores[selected]
        elite_badness = cp.empty((len(selected), 2 * initial.n), dtype=cp.int32)
        selected_scores = cp.empty(len(selected), dtype=cp.int32)
        badness_kernel(
            (len(selected),),
            (128,),
            (elite_p, elite_q, np.int32(initial.n), selected_scores, elite_badness),
        )
        elite_badness_host = cp.asnumpy(elite_badness)
        rank = rank_elites(elite_scores, elite_badness_host)
        rank_device = cp.asarray(rank)
        elite_p = elite_p[rank_device]
        elite_q = elite_q[rank_device]
        elite_badness = elite_badness[rank_device]
        elite_scores = elite_scores[rank]
        elite_badness_host = elite_badness_host[rank]
        generation_best = int(elite_scores[0])
        generation_key = elite_key(generation_best, elite_badness_host[0])
        improved = generation_key < best_key
        if improved:
            best_score = generation_best
            best_key = generation_key
            best_p = cp.asnumpy(elite_p[0])
            best_q = cp.asnumpy(elite_q[0])
            last_improvement = generation

        should_report = callback is not None and (improved or generation % max(1, report_every) == 0)
        if should_report:
            raw_p = cp.asnumpy(elite_p)
            raw_q = cp.asnumpy(elite_q)
            normalized = []
            for p, q in zip(raw_p, raw_q):
                state = _normalize_raw(p, q)
                normalized.append(np.column_stack((state.P, state.Q)))
            best_state = _normalize_raw(best_p, best_q)
            callback(
                GpuSearchSnapshot(
                    generation,
                    time.perf_counter() - started,
                    evaluations,
                    generation_best,
                    best_score,
                    tuple(map(int, elite_scores)),
                    np.asarray(normalized, dtype=np.int32),
                    best_state,
                )
            )
        if best_score == 0:
            state = _normalize_raw(best_p, best_q)
            ok_a, detail_a = verify_a(state.points(), state.n)
            ok_b, detail_b = verify_b(state.points(), state.n)
            if not (ok_a and ok_b):
                raise AssertionError(f"device GPU E=0 failed verification: A={detail_a}; B={detail_b}")
            break

    return GpuSearchResult(
        _normalize_raw(best_p, best_q),
        best_score,
        completed,
        evaluations,
        time.perf_counter() - started,
        seed,
    )
