from __future__ import annotations

import time
from functools import lru_cache
from random import Random
from typing import Callable

import numpy as np

from .gpu_device_search import _kernels, _normalize_raw, elite_key, rank_elites
from .gpu_search import GpuSearchResult, GpuSearchSnapshot
from .state import PairPermutations
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


ENUMERATE_SOURCE = r'''
extern "C" __global__
void enumerate_switches(
    const int* parent_p,
    const int* parent_q,
    const int* descriptors,
    const int candidate_count,
    const int n,
    int* child_p,
    int* child_q,
    unsigned char* valid
) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;
    const int parent = descriptors[3 * candidate];
    const int u = descriptors[3 * candidate + 1];
    const int v = descriptors[3 * candidate + 2];
    const int source = parent * n;
    const int target = candidate * n;
    for (int row = 0; row < n; ++row) {
        child_p[target + row] = parent_p[source + row];
        child_q[target + row] = parent_q[source + row];
    }
    valid[candidate] = 0;
    const int row_u = u >> 1;
    const int row_v = v >> 1;
    if (row_u == row_v) return;
    const int layer_u = u & 1;
    const int layer_v = v & 1;
    int* slot_u = layer_u ? child_q + target + row_u : child_p + target + row_u;
    int* slot_v = layer_v ? child_q + target + row_v : child_p + target + row_v;
    const int column_u = *slot_u;
    const int column_v = *slot_v;
    if (column_u == column_v) return;
    const int other_u = layer_u ? child_p[target + row_u] : child_q[target + row_u];
    const int other_v = layer_v ? child_p[target + row_v] : child_q[target + row_v];
    if (column_v == other_u || column_u == other_v) return;
    *slot_u = column_v;
    *slot_v = column_u;
    valid[candidate] = 1;
}
'''


@lru_cache(maxsize=1)
def _enumerate_kernel():
    cp, _, _, _ = _kernels()
    return cp.RawKernel(ENUMERATE_SOURCE, "enumerate_switches", options=("-std=c++11",))


def candidate_descriptors(badness: np.ndarray, *, global_anchors: bool = False) -> np.ndarray:
    """Describe every unordered switch touching a bad edge (or every switch)."""
    parent_count, edge_count = badness.shape
    descriptors: list[tuple[int, int, int]] = []
    for parent in range(parent_count):
        anchors = list(range(edge_count)) if global_anchors else list(map(int, np.flatnonzero(badness[parent])))
        anchor_set = set(anchors)
        for u in anchors:
            row_u = u >> 1
            for v in range(edge_count):
                if v == u or (v >> 1) == row_u:
                    continue
                if v in anchor_set and v < u:
                    continue
                descriptors.append((parent, u, v))
    return np.asarray(descriptors, dtype=np.int32).reshape((-1, 3))


def _signature(p: np.ndarray, q: np.ndarray) -> bytes:
    values = np.column_stack((p, q))
    values.sort(axis=1)
    return values.astype(np.int16, copy=False).tobytes()


def gpu_exhaustive_beam_search(
    initial: PairPermutations,
    *,
    seed: int,
    generations: int,
    elite_size: int,
    energy_window: int,
    initial_elites: np.ndarray | None = None,
    global_every: int = 25,
    report_every: int = 5,
    callback: Callable[[GpuSearchSnapshot], None] | None = None,
) -> GpuSearchResult:
    """Walk a diverse beam through complete bad-edge 2-switch neighborhoods."""
    initial.validate()
    if initial.n > 75:
        raise ValueError("CUDA kernels support n <= 75")
    if elite_size < 1 or generations < 1:
        raise ValueError("elite_size and generations must be positive")
    cp, _, badness_kernel, score_kernel = _kernels()
    enumerate_kernel = _enumerate_kernel()
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
    badness_kernel(
        (len(base),),
        (128,),
        (elite_p, elite_q, np.int32(initial.n), elite_scores_device, elite_badness),
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
    seen = {_signature(cp.asnumpy(elite_p[index]), cp.asnumpy(elite_q[index])) for index in range(len(base))}
    evaluations = len(base)
    started = time.perf_counter()
    completed = 0

    for generation in range(1, generations + 1):
        completed = generation
        badness_host = cp.asnumpy(elite_badness)
        descriptors = candidate_descriptors(
            badness_host,
            global_anchors=bool(global_every and generation % global_every == 0),
        )
        if not len(descriptors):
            break
        candidate_count = len(descriptors)
        descriptors_device = cp.asarray(descriptors)
        child_p = cp.empty((candidate_count, initial.n), dtype=cp.int32)
        child_q = cp.empty_like(child_p)
        valid_device = cp.empty(candidate_count, dtype=cp.uint8)
        enumerate_kernel(
            ((candidate_count + 255) // 256,),
            (256,),
            (
                elite_p,
                elite_q,
                descriptors_device,
                np.int32(candidate_count),
                np.int32(initial.n),
                child_p,
                child_q,
                valid_device,
            ),
        )
        scores_device = cp.empty(candidate_count, dtype=cp.int32)
        score_kernel((candidate_count,), (128,), (child_p, child_q, np.int32(initial.n), scores_device))
        scores = cp.asnumpy(scores_device)
        valid = cp.asnumpy(valid_device).astype(bool)
        scores[~valid] = 1 << 29
        evaluations += int(valid.sum())
        legal_order = np.flatnonzero(valid)
        legal_order = legal_order[np.argsort(scores[legal_order], kind="stable")]
        if not len(legal_order):
            break

        quality_count = max(1, elite_size // 4)
        quality_pool = list(map(int, legal_order[: min(len(legal_order), elite_size * 24)]))
        diverse_pool = list(map(int, np.flatnonzero(valid & (scores <= best_score + energy_window))))
        rng.shuffle(diverse_pool)
        prospects = []
        prospect_set: set[int] = set()
        for index in quality_pool + diverse_pool[: elite_size * 24]:
            if index not in prospect_set:
                prospects.append(index)
                prospect_set.add(index)
        prospect_array = cp.asarray(np.asarray(prospects, dtype=np.int32))
        prospect_p = cp.asnumpy(child_p[prospect_array])
        prospect_q = cp.asnumpy(child_q[prospect_array])
        lookup = {candidate: position for position, candidate in enumerate(prospects)}

        # The exact triple score often has a very broad plateau at E=10--13.
        # Rank the manageable prospect set by the same secondary badness
        # criteria as the retained beam, rather than accepting the arbitrary
        # descriptor order among equal-score children.
        prospect_scores_device = cp.empty(len(prospects), dtype=cp.int32)
        prospect_badness_device = cp.empty((len(prospects), 2 * initial.n), dtype=cp.int32)
        badness_kernel(
            (len(prospects),),
            (128,),
            (
                child_p[prospect_array],
                child_q[prospect_array],
                np.int32(initial.n),
                prospect_scores_device,
                prospect_badness_device,
            ),
        )
        prospect_badness = cp.asnumpy(prospect_badness_device)
        prospect_scores = scores[np.asarray(prospects, dtype=np.int32)]
        prospect_rank = rank_elites(prospect_scores, prospect_badness)
        quality_pool = [prospects[int(position)] for position in prospect_rank]

        selected: list[int] = []
        selected_signatures: set[bytes] = set()

        def retain(candidate: int, *, require_unseen: bool) -> bool:
            position = lookup.get(candidate)
            if position is None:
                return False
            signature = _signature(prospect_p[position], prospect_q[position])
            if signature in selected_signatures or (require_unseen and signature in seen):
                return False
            selected.append(candidate)
            selected_signatures.add(signature)
            return True

        for candidate in quality_pool:
            retain(candidate, require_unseen=True)
            if len(selected) == quality_count:
                break
        for candidate in diverse_pool:
            retain(candidate, require_unseen=True)
            if len(selected) == elite_size:
                break
        if len(selected) < elite_size:
            for candidate in quality_pool + diverse_pool:
                retain(candidate, require_unseen=False)
                if len(selected) == elite_size:
                    break
        if not selected:
            seen.clear()
            selected = quality_pool[:elite_size]
            selected_signatures = {
                _signature(prospect_p[lookup[candidate]], prospect_q[lookup[candidate]])
                for candidate in selected
                if candidate in lookup
            }
        seen.update(selected_signatures)
        if len(seen) > 500000:
            seen = set(selected_signatures)

        selected_device = cp.asarray(np.asarray(selected, dtype=np.int32))
        elite_p = child_p[selected_device]
        elite_q = child_q[selected_device]
        elite_scores = scores[selected]
        elite_badness = cp.empty((len(selected), 2 * initial.n), dtype=cp.int32)
        checked_scores = cp.empty(len(selected), dtype=cp.int32)
        badness_kernel(
            (len(selected),),
            (128,),
            (elite_p, elite_q, np.int32(initial.n), checked_scores, elite_badness),
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

        should_report = callback is not None and (improved or generation % max(1, report_every) == 0)
        if should_report:
            raw_p = cp.asnumpy(elite_p)
            raw_q = cp.asnumpy(elite_q)
            normalized = []
            for p, q in zip(raw_p, raw_q):
                state = _normalize_raw(p, q)
                normalized.append(np.column_stack((state.P, state.Q)))
            callback(
                GpuSearchSnapshot(
                    generation,
                    time.perf_counter() - started,
                    evaluations,
                    generation_best,
                    best_score,
                    tuple(map(int, elite_scores)),
                    np.asarray(normalized, dtype=np.int32),
                    _normalize_raw(best_p, best_q),
                )
            )
        if best_score == 0:
            state = _normalize_raw(best_p, best_q)
            ok_a, detail_a = verify_a(state.points(), state.n)
            ok_b, detail_b = verify_b(state.points(), state.n)
            if not (ok_a and ok_b):
                raise AssertionError(f"beam E=0 failed verification: A={detail_a}; B={detail_b}")
            break

    return GpuSearchResult(
        _normalize_raw(best_p, best_q),
        best_score,
        completed,
        evaluations,
        time.perf_counter() - started,
        seed,
    )
