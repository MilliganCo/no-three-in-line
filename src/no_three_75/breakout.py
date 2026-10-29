from __future__ import annotations

import math
import time
from dataclasses import dataclass
from random import Random
from typing import Callable

import numpy as np

from .accelerated import evaluate_two_switches, score_and_badness
from .conflicts import analyze_points
from .geometry import LineKey
from .search import SearchSnapshot
from .state import PairPermutations, points_to_permutations


@dataclass(frozen=True)
class BreakoutResult:
    state: PairPermutations
    score: int
    iterations: int
    elapsed_seconds: float
    moves_evaluated: int
    breakouts: int
    weighted_lines: int
    seed: int


def _slots(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(p)
    rows = np.repeat(np.arange(n, dtype=np.int32), 2)
    columns = np.empty(2 * n, dtype=np.int32)
    columns[0::2] = p
    columns[1::2] = q
    return rows, columns


def weighted_candidate_scores(
    p: np.ndarray,
    q: np.ndarray,
    selected_index: int,
    raw_scores: np.ndarray,
    weights: dict[LineKey, int],
) -> np.ndarray:
    weighted = raw_scores.astype(np.int64, copy=True)
    if not weights:
        return weighted
    active_weights = [(key, weight - 1) for key, weight in weights.items() if weight > 1]
    if not active_weights:
        return weighted
    rows, columns = _slots(p, q)
    coefficients = np.asarray([key for key, _ in active_weights], dtype=np.int64)
    extra_weights = np.asarray([weight for _, weight in active_weights], dtype=np.int64)[:, None]
    a = coefficients[:, 0:1]
    b = coefficients[:, 1:2]
    c = coefficients[:, 2:3]
    membership = a * rows[None, :] + b * columns[None, :] + c == 0
    occupancy = membership.sum(axis=1, keepdims=True)
    u = selected_index
    row_u = int(rows[u])
    column_u = int(columns[u])
    new_u = a * row_u + b * columns[None, :] + c == 0
    new_v = a * rows[None, :] + b * column_u + c == 0
    new_occupancy = occupancy - membership[:, u : u + 1] - membership + new_u + new_v
    triples = new_occupancy * (new_occupancy - 1) * (new_occupancy - 2) // 6
    triples[new_occupancy < 3] = 0
    weighted += (extra_weights * triples).sum(axis=0)
    return weighted


def weighted_state_score(p: np.ndarray, q: np.ndarray, raw_score: int, weights: dict[LineKey, int]) -> int:
    rows, columns = _slots(p, q)
    value = int(raw_score)
    for (coefficient_a, coefficient_b, coefficient_c), weight in weights.items():
        if weight <= 1:
            continue
        occupancy = int(np.count_nonzero(coefficient_a * rows + coefficient_b * columns + coefficient_c == 0))
        if occupancy >= 3:
            value += (weight - 1) * math.comb(occupancy, 3)
    return value


def breakout_search(
    initial: PairPermutations,
    *,
    seed: int,
    iterations: int,
    plateau: int,
    weight_increment: int,
    max_weighted_lines: int = 128,
    checkpoint: Callable[[SearchSnapshot], None] | None = None,
    move_evaluator: Callable[[np.ndarray, np.ndarray, int, int, np.ndarray], np.ndarray] = evaluate_two_switches,
) -> BreakoutResult:
    initial.validate()
    rng = Random(seed)
    n = initial.n
    p = np.asarray(initial.P, dtype=np.int32).copy()
    q = np.asarray(initial.Q, dtype=np.int32).copy()
    raw_score, badness = score_and_badness(p, q)
    best_score = int(raw_score)
    best_p = p.copy()
    best_q = q.copy()
    weights: dict[LineKey, int] = {}
    weighted_score = int(raw_score)
    last_raw_improvement = 0
    moves_evaluated = 0
    breakouts = 0
    started = time.perf_counter()

    for iteration in range(1, iterations + 1):
        weighted_badness = badness.astype(np.int64, copy=True)
        if weights:
            rows, columns = _slots(p, q)
            active_weights = [(key, weight - 1) for key, weight in weights.items() if weight > 1]
            if active_weights:
                coefficients = np.asarray([key for key, _ in active_weights], dtype=np.int64)
                extras = np.asarray([weight for _, weight in active_weights], dtype=np.int64)
                membership = (
                    coefficients[:, 0:1] * rows[None, :]
                    + coefficients[:, 1:2] * columns[None, :]
                    + coefficients[:, 2:3]
                    == 0
                )
                occupancy = membership.sum(axis=1)
                contributions = extras * np.maximum(0, (occupancy - 1) * (occupancy - 2) // 2)
                weighted_badness += (contributions[:, None] * membership).sum(axis=0)
        if rng.random() < 0.05 or int(weighted_badness.max()) == 0:
            selected = rng.randrange(2 * n)
        else:
            maximum = int(weighted_badness.max())
            pool = [index for index in range(2 * n) if int(weighted_badness[index]) == maximum]
            selected = rng.choice(pool)

        raw_scores = move_evaluator(p, q, selected, int(raw_score), badness)
        weighted_scores = weighted_candidate_scores(p, q, selected, raw_scores, weights)
        legal = np.flatnonzero(weighted_scores < (1 << 29))
        moves_evaluated += len(legal)
        if len(legal):
            minimum = int(weighted_scores[legal].min())
            best_moves = [int(index) for index in legal if int(weighted_scores[index]) == minimum]
            chosen = rng.choice(best_moves)
            if minimum <= weighted_score:
                rows, columns = _slots(p, q)
                row_a, layer_a = divmod(selected, 2)
                row_b, layer_b = divmod(chosen, 2)
                point_a = (row_a, int(columns[selected]))
                point_b = (row_b, int(columns[chosen]))
                points = [(int(rows[index]), int(columns[index])) for index in range(2 * n)]
                points.remove(point_a)
                points.remove(point_b)
                points.extend(((row_a, point_b[1]), (row_b, point_a[1])))
                state = points_to_permutations(points, n)
                p = np.asarray(state.P, dtype=np.int32)
                q = np.asarray(state.Q, dtype=np.int32)
                raw_score, badness = score_and_badness(p, q)
                weighted_score = weighted_state_score(p, q, int(raw_score), weights)
                if weighted_score != minimum:
                    raise AssertionError(f"weighted delta {minimum} != recomputed score {weighted_score}")
                if int(raw_score) < best_score:
                    best_score = int(raw_score)
                    best_p = p.copy()
                    best_q = q.copy()
                    last_raw_improvement = iteration
                    if checkpoint is not None:
                        checkpoint(
                            SearchSnapshot(
                                iteration,
                                time.perf_counter() - started,
                                int(raw_score),
                                best_score,
                                layer_a,
                                row_a,
                                row_b,
                                layer_b,
                                PairPermutations(tuple(map(int, p)), tuple(map(int, q))),
                            )
                        )
                    if best_score == 0:
                        break

        if iteration - last_raw_improvement >= plateau:
            metrics = analyze_points(
                [(row, int(p[row])) for row in range(n)] + [(row, int(q[row])) for row in range(n)]
            )
            for line in metrics.lines:
                weights[line.key] = weights.get(line.key, 1) + weight_increment
            if breakouts and breakouts % 8 == 0:
                active_keys = {line.key for line in metrics.lines}
                weights = {
                    key: weight - (key not in active_keys)
                    for key, weight in weights.items()
                    if weight - (key not in active_keys) > 1
                }
            if len(weights) > max_weighted_lines:
                ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:max_weighted_lines]
                weights = dict(ranked)
            weighted_score = weighted_state_score(p, q, int(raw_score), weights)
            breakouts += 1
            last_raw_improvement = iteration

    return BreakoutResult(
        PairPermutations(tuple(map(int, best_p)), tuple(map(int, best_q))),
        best_score,
        iteration if iterations else 0,
        time.perf_counter() - started,
        moves_evaluated,
        breakouts,
        len(weights),
        seed,
    )
