from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from .accelerated import score_and_badness
from .gpu import score_population_gpu
from .state import PairPermutations, points_to_permutations


@dataclass(frozen=True)
class FourCycleResult:
    state: PairPermutations
    score: int
    anchor_index: int
    evaluated: int
    legal: int
    elapsed_seconds: float


@dataclass(frozen=True)
class DoubleSwitchResult:
    state: PairPermutations
    score: int
    anchor_index: int
    evaluated: int
    legal: int
    elapsed_seconds: float


def _values_state(values: np.ndarray) -> PairPermutations:
    n = values.shape[0]
    points = [(row, int(values[row, layer])) for row in range(n) for layer in range(2)]
    return points_to_permutations(points, n)


def _score_mapping_batch(
    base: np.ndarray,
    targets: np.ndarray,
    sources: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_base = base.reshape(-1)
    new_columns = flat_base[sources]
    other_slots = np.bitwise_xor(targets, 1)
    legal_mask = np.all(new_columns != flat_base[other_slots], axis=1)
    legal_targets = targets[legal_mask]
    legal_sources = sources[legal_mask]
    legal_columns = new_columns[legal_mask]
    if len(legal_targets) == 0:
        return np.empty(0, dtype=np.int32), legal_targets, legal_sources
    candidates = np.broadcast_to(base, (len(legal_targets),) + base.shape).copy()
    candidate_flat = candidates.reshape(len(legal_targets), -1)
    candidate_flat[np.arange(len(legal_targets))[:, None], legal_targets] = legal_columns
    scores = score_population_gpu(
        np.ascontiguousarray(candidates[:, :, 0]),
        np.ascontiguousarray(candidates[:, :, 1]),
    )
    return scores, legal_targets, legal_sources


def _score_cycle_batch(base: np.ndarray, moves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sources = moves[:, (1, 2, 3, 0)]
    scores, legal_moves, _ = _score_mapping_batch(base, moves, sources)
    return scores, legal_moves


def best_four_cycle(
    state: PairPermutations,
    anchor_index: int,
    *,
    batch_size: int = 32768,
) -> FourCycleResult:
    state.validate()
    n = state.n
    if not (0 <= anchor_index < 2 * n):
        raise ValueError("anchor index outside point slots")
    base = np.column_stack((state.P, state.Q)).astype(np.int32)
    base_score = int(score_and_badness(np.asarray(state.P), np.asarray(state.Q))[0])
    best_score = base_score
    best_values = base.copy()
    anchor_row = anchor_index // 2
    other_rows = [row for row in range(n) if row != anchor_row]
    buffer = np.empty((batch_size, 4), dtype=np.int16)
    buffered = 0
    evaluated = 0
    legal = 0
    started = time.perf_counter()

    def evaluate(count: int) -> None:
        nonlocal best_score, best_values, evaluated, legal
        raw_moves = buffer[:count].astype(np.int32, copy=False)
        scores, legal_moves = _score_cycle_batch(base, raw_moves)
        evaluated += count
        legal += len(legal_moves)
        if len(scores) == 0:
            return
        index = int(np.argmin(scores))
        score = int(scores[index])
        if score >= best_score:
            return
        move = legal_moves[index]
        flat_base = base.reshape(-1)
        changed = base.copy().reshape(-1)
        changed[move] = flat_base[move[np.asarray((1, 2, 3, 0))]]
        candidate = changed.reshape(n, 2)
        exact = int(score_and_badness(candidate[:, 0], candidate[:, 1])[0])
        if exact != score:
            raise AssertionError(f"GPU four-cycle score {score} != CPU score {exact}")
        best_score = score
        best_values = candidate.copy()

    for rows in itertools.combinations(other_rows, 3):
        for layers in itertools.product((0, 1), repeat=3):
            slots = tuple(2 * row + layer for row, layer in zip(rows, layers))
            for ordering in itertools.permutations(slots):
                buffer[buffered] = (anchor_index,) + ordering
                buffered += 1
                if buffered == batch_size:
                    evaluate(buffered)
                    buffered = 0
    if buffered:
        evaluate(buffered)
    return FourCycleResult(
        _values_state(best_values),
        best_score,
        anchor_index,
        evaluated,
        legal,
        time.perf_counter() - started,
    )


def bad_anchor_indices(state: PairPermutations, limit: int) -> list[int]:
    p = np.asarray(state.P, dtype=np.int32)
    q = np.asarray(state.Q, dtype=np.int32)
    _, badness = score_and_badness(p, q)
    return sorted(range(2 * state.n), key=lambda index: (-int(badness[index]), index))[:limit]


def best_double_switch(
    state: PairPermutations,
    anchor_index: int,
    *,
    batch_size: int = 32768,
) -> DoubleSwitchResult:
    state.validate()
    n = state.n
    if not (0 <= anchor_index < 2 * n):
        raise ValueError("anchor index outside point slots")
    base = np.column_stack((state.P, state.Q)).astype(np.int32)
    base_score = int(score_and_badness(np.asarray(state.P), np.asarray(state.Q))[0])
    best_score = base_score
    best_values = base.copy()
    anchor_row = anchor_index // 2
    other_rows = [row for row in range(n) if row != anchor_row]
    target_buffer = np.empty((batch_size, 4), dtype=np.int16)
    source_buffer = np.empty((batch_size, 4), dtype=np.int16)
    buffered = 0
    evaluated = 0
    legal = 0
    started = time.perf_counter()

    def evaluate(count: int) -> None:
        nonlocal best_score, best_values, evaluated, legal
        targets = target_buffer[:count].astype(np.int32, copy=False)
        sources = source_buffer[:count].astype(np.int32, copy=False)
        scores, legal_targets, legal_sources = _score_mapping_batch(base, targets, sources)
        evaluated += count
        legal += len(legal_targets)
        if len(scores) == 0:
            return
        index = int(np.argmin(scores))
        score = int(scores[index])
        if score >= best_score:
            return
        targets_best = legal_targets[index]
        sources_best = legal_sources[index]
        changed = base.copy().reshape(-1)
        flat_base = base.reshape(-1)
        changed[targets_best] = flat_base[sources_best]
        candidate = changed.reshape(n, 2)
        exact = int(score_and_badness(candidate[:, 0], candidate[:, 1])[0])
        if exact != score:
            raise AssertionError(f"GPU double-switch score {score} != CPU score {exact}")
        best_score = score
        best_values = candidate.copy()

    for rows in itertools.combinations(other_rows, 3):
        for layers in itertools.product((0, 1), repeat=3):
            a, b, c = (2 * row + layer for row, layer in zip(rows, layers))
            targets = (anchor_index, a, b, c)
            for sources in ((a, anchor_index, c, b), (b, c, anchor_index, a), (c, b, a, anchor_index)):
                target_buffer[buffered] = targets
                source_buffer[buffered] = sources
                buffered += 1
                if buffered == batch_size:
                    evaluate(buffered)
                    buffered = 0
    if buffered:
        evaluate(buffered)
    return DoubleSwitchResult(
        _values_state(best_values),
        best_score,
        anchor_index,
        evaluated,
        legal,
        time.perf_counter() - started,
    )
