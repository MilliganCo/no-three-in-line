from __future__ import annotations

import math
import time
from dataclasses import dataclass
from random import Random
from typing import Callable

import numpy as np

from .accelerated import evaluate_three_cycles, evaluate_two_switches, score_and_badness
from .state import PairPermutations, points_to_permutations, random_state


@dataclass(frozen=True)
class SearchSnapshot:
    iteration: int
    elapsed_seconds: float
    score: int
    best_score: int
    layer: int
    row_a: int
    row_b: int
    partner_layer: int
    state: PairPermutations


@dataclass(frozen=True)
class SearchResult:
    state: PairPermutations
    score: int
    iterations: int
    elapsed_seconds: float
    moves_evaluated: int
    seed: int


def legal_three_cycle_candidates(p: np.ndarray, q: np.ndarray, selected_index: int) -> np.ndarray:
    n = len(p)
    columns = np.empty(2 * n, dtype=np.int32)
    for row in range(n):
        columns[2 * row] = p[row]
        columns[2 * row + 1] = q[row]
    selected_row = selected_index // 2
    candidates: list[tuple[int, int, int, int]] = []
    for v in range(2 * n - 1):
        row_v = v // 2
        if row_v == selected_row:
            continue
        for w in range(v + 1, 2 * n):
            row_w = w // 2
            if row_w == selected_row or row_w == row_v:
                continue
            if len({int(columns[selected_index]), int(columns[v]), int(columns[w])}) != 3:
                continue
            other_u = 2 * selected_row + (1 - (selected_index & 1))
            other_v = 2 * row_v + (1 - (v & 1))
            other_w = 2 * row_w + (1 - (w & 1))
            # u <- v, v <- w, w <- u
            if columns[v] != columns[other_u] and columns[w] != columns[other_v] and columns[selected_index] != columns[other_w]:
                candidates.append((selected_index, v, w, 0))
            # u <- w, w <- v, v <- u
            if columns[w] != columns[other_u] and columns[v] != columns[other_w] and columns[selected_index] != columns[other_v]:
                candidates.append((selected_index, v, w, 1))
    return np.asarray(candidates, dtype=np.int32).reshape((-1, 4))


def apply_three_cycle(p: np.ndarray, q: np.ndarray, candidate: np.ndarray) -> PairPermutations:
    n = len(p)
    u, v, w, orientation = map(int, candidate)
    points = [(row, int(p[row])) for row in range(n)] + [(row, int(q[row])) for row in range(n)]
    indices = (u, v, w)
    old = [(index // 2, int(p[index // 2] if index % 2 == 0 else q[index // 2])) for index in indices]
    for point in old:
        points.remove(point)
    if orientation == 0:
        new_columns = (old[1][1], old[2][1], old[0][1])
    else:
        new_columns = (old[2][1], old[0][1], old[1][1])
    points.extend((old[index][0], new_columns[index]) for index in range(3))
    return points_to_permutations(points, n)


def local_search(
    n: int,
    *,
    seed: int,
    iterations: int,
    initial: PairPermutations | None = None,
    tabu_tenure: int = 11,
    plateau: int = 800,
    exploration_probability: float = 0.04,
    checkpoint: Callable[[SearchSnapshot], None] | None = None,
    move_evaluator: Callable[[np.ndarray, np.ndarray, int, int, np.ndarray], np.ndarray] = evaluate_two_switches,
) -> SearchResult:
    rng = Random(seed)
    state = initial if initial is not None else random_state(n, rng)
    state.validate()
    p = np.asarray(state.P, dtype=np.int32).copy()
    q = np.asarray(state.Q, dtype=np.int32).copy()
    current_score, badness = score_and_badness(p, q)
    best_p = p.copy()
    best_q = q.copy()
    best_score = int(current_score)
    tabu_until: dict[tuple[int, int, int], int] = {}
    last_improvement = 0
    moves_evaluated = 0
    started = time.perf_counter()

    for iteration in range(1, iterations + 1):
        weights = badness.astype(np.float64) + 0.05
        if rng.random() < exploration_probability or float(weights.sum()) == 0.0:
            selected_index = rng.randrange(2 * n)
        else:
            threshold = rng.random() * float(weights.sum())
            cumulative = 0.0
            selected_index = 0
            for index, weight in enumerate(weights):
                cumulative += float(weight)
                if cumulative >= threshold:
                    selected_index = index
                    break
        selected_row, layer = divmod(selected_index, 2)
        candidate_scores = move_evaluator(p, q, selected_index, current_score, badness)
        legal = [index for index in range(2 * n) if candidate_scores[index] < (1 << 29)]
        moves_evaluated += len(legal)
        if not legal:
            continue
        legal.sort(key=lambda index: (int(candidate_scores[index]), rng.random()))
        chosen = None
        for partner_index in legal:
            key = (min(selected_index, partner_index), max(selected_index, partner_index), 2)
            aspiration = int(candidate_scores[partner_index]) < best_score
            if tabu_until.get(key, -1) <= iteration or aspiration:
                chosen = partner_index
                break
        if chosen is None:
            chosen = legal[0]

        proposed_score = int(candidate_scores[chosen])
        temperature = max(0.05, 1.5 * (1.0 - min(1.0, (iteration - last_improvement) / max(1, plateau))))
        delta = proposed_score - int(current_score)
        accept = delta <= 0 or rng.random() < math.exp(-delta / temperature)

        if accept:
            partner_row, partner_layer = divmod(chosen, 2)
            points = [(row, int(p[row])) for row in range(n)] + [(row, int(q[row])) for row in range(n)]
            selected_point = (selected_row, int(p[selected_row] if layer == 0 else q[selected_row]))
            partner_point = (partner_row, int(p[partner_row] if partner_layer == 0 else q[partner_row]))
            points.remove(selected_point)
            points.remove(partner_point)
            points.append((selected_row, partner_point[1]))
            points.append((partner_row, selected_point[1]))
            recolored = points_to_permutations(points, n)
            p = np.asarray(recolored.P, dtype=np.int32)
            q = np.asarray(recolored.Q, dtype=np.int32)
            current_score, badness = score_and_badness(p, q)
            if int(current_score) != proposed_score:
                raise AssertionError(f"incremental score {proposed_score} != full score {current_score}")
            key = (min(selected_index, chosen), max(selected_index, chosen), 2)
            tabu_until[key] = iteration + tabu_tenure

            if int(current_score) < best_score:
                best_score = int(current_score)
                best_p = p.copy()
                best_q = q.copy()
                last_improvement = iteration
                if checkpoint is not None:
                    checkpoint(
                        SearchSnapshot(
                            iteration,
                            time.perf_counter() - started,
                            int(current_score),
                            best_score,
                            layer,
                            selected_row,
                            partner_row,
                            partner_layer,
                            PairPermutations(tuple(map(int, p)), tuple(map(int, q))),
                        )
                    )
                if best_score == 0:
                    break

        if iteration - last_improvement >= plateau:
            # Exhaustive three-edge cycle around a maximally bad point. This changes
            # three rows at once and crosses 2-switch local minima while preserving
            # every row/column degree exactly.
            maximum_badness = int(badness.max())
            bad_indices = [index for index in range(2 * n) if int(badness[index]) == maximum_badness]
            cycle_anchor = rng.choice(bad_indices) if bad_indices else rng.randrange(2 * n)
            cycle_candidates = legal_three_cycle_candidates(p, q, cycle_anchor)
            used_cycle = False
            if len(cycle_candidates):
                cycle_scores = evaluate_three_cycles(p, q, cycle_candidates, current_score, badness)
                moves_evaluated += len(cycle_candidates)
                cycle_index = int(np.argmin(cycle_scores))
                cycle_score = int(cycle_scores[cycle_index])
                if cycle_score <= int(current_score) + 3:
                    recolored = apply_three_cycle(p, q, cycle_candidates[cycle_index])
                    p = np.asarray(recolored.P, dtype=np.int32)
                    q = np.asarray(recolored.Q, dtype=np.int32)
                    current_score, badness = score_and_badness(p, q)
                    if int(current_score) != cycle_score:
                        raise AssertionError(f"three-cycle score {cycle_score} != full score {current_score}")
                    used_cycle = True
                    if int(current_score) < best_score:
                        best_score = int(current_score)
                        best_p = p.copy()
                        best_q = q.copy()
                        if checkpoint is not None:
                            cycle = cycle_candidates[cycle_index]
                            partner_row = int(cycle[1]) // 2
                            checkpoint(
                                SearchSnapshot(
                                    iteration,
                                    time.perf_counter() - started,
                                    int(current_score),
                                    best_score,
                                    int(cycle[0]) % 2,
                                    int(cycle[0]) // 2,
                                    partner_row,
                                    int(cycle[1]) % 2,
                                    PairPermutations(tuple(map(int, p)), tuple(map(int, q))),
                                )
                            )
            if used_cycle:
                last_improvement = iteration
                if best_score == 0:
                    break
                continue

            perturbations = 2 + rng.randrange(7)
            for _ in range(perturbations):
                for _attempt in range(100):
                    point_a = rng.randrange(2 * n)
                    point_b = rng.randrange(2 * n)
                    row_a, layer_a = divmod(point_a, 2)
                    row_b, layer_b = divmod(point_b, 2)
                    if row_a == row_b:
                        continue
                    column_a = int(p[row_a] if layer_a == 0 else q[row_a])
                    column_b = int(p[row_b] if layer_b == 0 else q[row_b])
                    other_a = int(q[row_a] if layer_a == 0 else p[row_a])
                    other_b = int(q[row_b] if layer_b == 0 else p[row_b])
                    if column_b == other_a or column_a == other_b:
                        continue
                    points = [(row, int(p[row])) for row in range(n)] + [(row, int(q[row])) for row in range(n)]
                    points.remove((row_a, column_a))
                    points.remove((row_b, column_b))
                    points.append((row_a, column_b))
                    points.append((row_b, column_a))
                    recolored = points_to_permutations(points, n)
                    p = np.asarray(recolored.P, dtype=np.int32)
                    q = np.asarray(recolored.Q, dtype=np.int32)
                    break
            current_score, badness = score_and_badness(p, q)
            last_improvement = iteration

    elapsed = time.perf_counter() - started
    return SearchResult(
        PairPermutations(tuple(map(int, best_p)), tuple(map(int, best_q))),
        best_score,
        iteration if iterations else 0,
        elapsed,
        moves_evaluated,
        seed,
    )
