from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable

import numpy as np

from .accelerated import score_and_badness
from .geometry import d4_variants
from .gpu import score_population_gpu
from .state import PairPermutations, expand_state, matching_decompositions


@dataclass(frozen=True)
class ExpansionStep:
    inserted_row: int
    inserted_column: int
    break_p: int | None
    break_q: int | None


@dataclass(frozen=True)
class MultiExpansionRecord:
    score: int
    source_variant: int
    decomposition: int
    steps: tuple[ExpansionStep, ...]
    state: PairPermutations


@dataclass(frozen=True)
class MultiExpansionExperiment:
    evaluated: int
    histogram: dict[int, int]
    best: tuple[MultiExpansionRecord, ...]


def _random_breaks(state: PairPermutations, rng: Random) -> tuple[int | None, int | None]:
    n = state.n
    mode = rng.randrange(3)
    if mode == 0:
        return None, rng.randrange(n)
    if mode == 1:
        return rng.randrange(n), None
    for _ in range(100):
        break_p = rng.randrange(n)
        break_q = rng.randrange(n)
        if break_p != break_q and state.P[break_p] != state.Q[break_q]:
            return break_p, break_q
    raise RuntimeError("could not sample compatible expansion breaks")


def _score_batch(states: list[PairPermutations], backend: str) -> np.ndarray:
    p_batch = np.asarray([state.P for state in states], dtype=np.int32)
    q_batch = np.asarray([state.Q for state in states], dtype=np.int32)
    if backend == "gpu":
        return score_population_gpu(p_batch, q_batch)
    if backend == "cpu":
        return np.asarray(
            [score_and_badness(p, q)[0] for p, q in zip(p_batch, q_batch)],
            dtype=np.int32,
        )
    raise ValueError("backend must be cpu or gpu")


def sample_multistep_expansions(
    source: PairPermutations,
    target_n: int,
    *,
    samples: int,
    keep: int,
    seed: int,
    batch_size: int,
    backend: str,
    allow_interior_insertions: bool,
    callback: Callable[[int, int], None] | None = None,
) -> MultiExpansionExperiment:
    if target_n <= source.n:
        raise ValueError("target_n must exceed the source size")
    if samples < 1 or keep < 1 or batch_size < 1:
        raise ValueError("samples, keep, and batch_size must be positive")
    rng = Random(seed)
    starting_states: list[tuple[int, int, PairPermutations]] = []
    for variant_index, points in enumerate(d4_variants(source.points(), source.n)):
        for decomposition_index, state in enumerate(matching_decompositions(points, source.n)):
            starting_states.append((variant_index, decomposition_index, state))

    histogram: dict[int, int] = {}
    retained: list[MultiExpansionRecord] = []
    evaluated = 0
    while evaluated < samples:
        count = min(batch_size, samples - evaluated)
        states: list[PairPermutations] = []
        provenance: list[tuple[int, int, tuple[ExpansionStep, ...]]] = []
        for _ in range(count):
            variant_index, decomposition_index, state = starting_states[rng.randrange(len(starting_states))]
            steps: list[ExpansionStep] = []
            while state.n < target_n:
                inserted_row = rng.randrange(state.n + 1) if allow_interior_insertions else state.n
                inserted_column = rng.randrange(state.n + 1) if allow_interior_insertions else state.n
                break_p, break_q = _random_breaks(state, rng)
                state = expand_state(
                    state,
                    inserted_row,
                    inserted_column,
                    break_p=break_p,
                    break_q=break_q,
                )
                steps.append(ExpansionStep(inserted_row, inserted_column, break_p, break_q))
            states.append(state)
            provenance.append((variant_index, decomposition_index, tuple(steps)))

        scores = _score_batch(states, backend)
        for state, score_value, (variant_index, decomposition_index, steps) in zip(states, scores, provenance):
            score = int(score_value)
            histogram[score] = histogram.get(score, 0) + 1
            record = MultiExpansionRecord(score, variant_index, decomposition_index, steps, state)
            if len(retained) < keep:
                retained.append(record)
            else:
                worst_index = max(range(len(retained)), key=lambda index: retained[index].score)
                if score < retained[worst_index].score:
                    retained[worst_index] = record
        evaluated += count
        if callback:
            callback(evaluated, min(record.score for record in retained))

    retained.sort(
        key=lambda record: (
            record.score,
            record.source_variant,
            record.decomposition,
            tuple(
                (
                    step.inserted_row,
                    step.inserted_column,
                    -1 if step.break_p is None else step.break_p,
                    -1 if step.break_q is None else step.break_q,
                )
                for step in record.steps
            ),
        )
    )
    return MultiExpansionExperiment(evaluated, dict(sorted(histogram.items())), tuple(retained))
