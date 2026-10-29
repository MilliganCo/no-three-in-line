from __future__ import annotations

import time
from dataclasses import dataclass
from random import Random
from typing import Callable

import numpy as np

from .gpu import score_population_gpu
from .state import PairPermutations, points_to_permutations, repair_collisions
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b


@dataclass(frozen=True)
class GpuSearchSnapshot:
    generation: int
    elapsed_seconds: float
    evaluations: int
    generation_best: int
    best_score: int
    elite_scores: tuple[int, ...]
    elite_values: np.ndarray
    state: PairPermutations


@dataclass(frozen=True)
class GpuSearchResult:
    state: PairPermutations
    score: int
    generations: int
    evaluations: int
    elapsed_seconds: float
    seed: int


def _state_values(state: PairPermutations) -> np.ndarray:
    values = np.empty((state.n, 2), dtype=np.int32)
    values[:, 0] = state.P
    values[:, 1] = state.Q
    return values


def _values_state(values: np.ndarray) -> PairPermutations:
    points = [
        (row, int(values[row, layer]))
        for row in range(values.shape[0])
        for layer in range(2)
    ]
    return points_to_permutations(points, values.shape[0])


def mutate_values(values: np.ndarray, rng: Random, strength: int) -> np.ndarray:
    """Apply legal edge 2-switches; row and column degrees remain exactly two."""
    child = values.copy()
    n = child.shape[0]
    for _ in range(strength):
        for _attempt in range(100):
            row_a = rng.randrange(n)
            row_b = rng.randrange(n - 1)
            if row_b >= row_a:
                row_b += 1
            layer_a = rng.randrange(2)
            layer_b = rng.randrange(2)
            column_a = int(child[row_a, layer_a])
            column_b = int(child[row_b, layer_b])
            if column_a == column_b:
                continue
            if column_b == int(child[row_a, 1 - layer_a]):
                continue
            if column_a == int(child[row_b, 1 - layer_b]):
                continue
            child[row_a, layer_a], child[row_b, layer_b] = column_b, column_a
            break
        else:
            raise RuntimeError("could not generate a legal degree-preserving mutation")
    return child


def cycle_crossover_permutation(left: np.ndarray, right: np.ndarray, rng: Random) -> np.ndarray:
    n = len(left)
    if sorted(map(int, left)) != list(range(n)) or sorted(map(int, right)) != list(range(n)):
        raise ValueError("cycle crossover parents must be permutations")
    inverse_left = np.empty(n, dtype=np.int32)
    for index, value in enumerate(left):
        inverse_left[int(value)] = index
    child = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)
    for start in range(n):
        if visited[start]:
            continue
        cycle: list[int] = []
        index = start
        while not visited[index]:
            visited[index] = True
            cycle.append(index)
            index = int(inverse_left[int(right[index])])
        source = left if rng.random() < 0.5 else right
        for position in cycle:
            child[position] = source[position]
    if sorted(map(int, child)) != list(range(n)):
        raise AssertionError("cycle crossover did not produce a permutation")
    return child


def crossover_values(left: np.ndarray, right: np.ndarray, rng: Random) -> np.ndarray:
    state = PairPermutations(
        tuple(map(int, cycle_crossover_permutation(left[:, 0], right[:, 0], rng))),
        tuple(map(int, cycle_crossover_permutation(left[:, 1], right[:, 1], rng))),
    )
    return _state_values(repair_collisions(state))


def _signature(values: np.ndarray) -> bytes:
    return np.sort(values, axis=1).astype(np.int16, copy=False).tobytes()


def gpu_population_search(
    initial: PairPermutations,
    *,
    seed: int,
    generations: int,
    population: int,
    elite_size: int,
    max_strength: int,
    energy_window: int = 12,
    crossover_probability: float = 0.0,
    initial_elites: np.ndarray | None = None,
    callback: Callable[[GpuSearchSnapshot], None] | None = None,
) -> GpuSearchResult:
    if not (1 <= elite_size <= population):
        raise ValueError("elite_size must be between 1 and population")
    if max_strength < 1:
        raise ValueError("max_strength must be positive")
    initial.validate()
    rng = Random(seed)
    initial_values = _state_values(initial)
    if initial_elites is None:
        elite_array = initial_values[None, :, :]
    else:
        elite_array = np.ascontiguousarray(initial_elites, dtype=np.int32)
        if elite_array.ndim != 3 or elite_array.shape[1:] != (initial.n, 2):
            raise ValueError("resume elite array must have shape (population,n,2)")
        for values in elite_array:
            _values_state(values).validate()
        elite_array = np.concatenate((initial_values[None, :, :], elite_array), axis=0)
    starting_scores = score_population_gpu(
        np.ascontiguousarray(elite_array[:, :, 0]),
        np.ascontiguousarray(elite_array[:, :, 1]),
    )
    starting_order = np.argsort(starting_scores, kind="stable")[:elite_size]
    elites = [elite_array[int(index)].copy() for index in starting_order]
    elite_scores = [int(starting_scores[int(index)]) for index in starting_order]
    best_values = elites[0].copy()
    best_score = elite_scores[0]
    initial_score = best_score
    evaluations = len(elite_array)
    started = time.perf_counter()

    if initial_score == 0:
        return GpuSearchResult(initial, 0, 0, evaluations, time.perf_counter() - started, seed)

    completed_generations = 0
    last_improvement = 0
    for generation in range(1, generations + 1):
        completed_generations = generation
        candidates = np.empty((population, initial.n, 2), dtype=np.int32)
        retained_count = min(len(elites), population)
        for index in range(retained_count):
            candidates[index] = elites[index]
        active_strength = min(max_strength, 3 + (generation - last_improvement) // 8)
        quality_pool = min(len(elites), max(1, elite_size // 4))
        normalized_elites = None
        if crossover_probability > 0 and len(elites) > 1:
            normalized_elites = [_state_values(_values_state(values)) for values in elites]
        for index in range(retained_count, population):
            if normalized_elites is not None and rng.random() < crossover_probability:
                left_index = rng.randrange(len(normalized_elites))
                right_index = rng.randrange(len(normalized_elites) - 1)
                if right_index >= left_index:
                    right_index += 1
                parent = crossover_values(normalized_elites[left_index], normalized_elites[right_index], rng)
            elif rng.random() < 0.45:
                parent = elites[rng.randrange(quality_pool)]
            else:
                parent = elites[rng.randrange(len(elites))]
            if rng.random() < 0.18:
                strength = rng.randint(1, max_strength)
            else:
                strength = min(active_strength, 1 + int(rng.expovariate(0.72)))
            candidates[index] = mutate_values(parent, rng, strength)

        p_batch = np.ascontiguousarray(candidates[:, :, 0])
        q_batch = np.ascontiguousarray(candidates[:, :, 1])
        scores = score_population_gpu(p_batch, q_batch)
        evaluations += population
        order = np.argsort(scores, kind="stable")
        next_elites: list[np.ndarray] = []
        next_scores: list[int] = []
        seen: set[bytes] = set()

        def retain(candidate_index: int) -> bool:
            signature = _signature(candidates[candidate_index])
            if signature in seen:
                return False
            seen.add(signature)
            next_elites.append(candidates[candidate_index].copy())
            next_scores.append(int(scores[candidate_index]))
            return True

        quality_count = max(1, elite_size // 4)
        for raw_index in order:
            candidate_index = int(raw_index)
            retain(candidate_index)
            if len(next_elites) == quality_count:
                break
        diverse = [int(index) for index in np.flatnonzero(scores <= best_score + energy_window)]
        rng.shuffle(diverse)
        for candidate_index in diverse:
            retain(candidate_index)
            if len(next_elites) == elite_size:
                break
        if len(next_elites) < elite_size:
            for raw_index in order:
                retain(int(raw_index))
                if len(next_elites) == elite_size:
                    break
        elites = next_elites
        elite_scores = next_scores
        generation_best = elite_scores[0]
        improved = generation_best < best_score
        if improved:
            best_score = generation_best
            best_values = elites[0].copy()
            last_improvement = generation
        if callback is not None:
            callback(
                GpuSearchSnapshot(
                    generation,
                    time.perf_counter() - started,
                    evaluations,
                    generation_best,
                    best_score,
                    tuple(elite_scores),
                    np.asarray(elites, dtype=np.int32),
                    _values_state(best_values),
                )
            )
        if best_score == 0:
            state = _values_state(best_values)
            points = state.points()
            ok_a, detail_a = verify_a(points, state.n)
            ok_b, detail_b = verify_b(points, state.n)
            if not (ok_a and ok_b):
                raise AssertionError(f"GPU E=0 failed verification: A={detail_a}; B={detail_b}")
            break

    return GpuSearchResult(
        _values_state(best_values),
        best_score,
        completed_generations,
        evaluations,
        time.perf_counter() - started,
        seed,
    )
