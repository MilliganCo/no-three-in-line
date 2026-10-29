from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .accelerated import score_and_badness
from .geometry import d4_variants
from .state import PairPermutations, expand_state, matching_decompositions, project_state


@dataclass(frozen=True)
class SeedRecord:
    score: int
    removed_row: int
    removed_column: int
    variant: int
    decomposition: int
    state: PairPermutations


@dataclass(frozen=True)
class SeedExperiment:
    evaluated: int
    unique: int
    histogram: dict[int, int]
    best: tuple[SeedRecord, ...]


@dataclass(frozen=True)
class ExpansionSeedRecord:
    score: int
    break_p: int | None
    break_q: int | None
    variant: int
    decomposition: int
    state: PairPermutations


@dataclass(frozen=True)
class ExpansionSeedExperiment:
    evaluated: int
    unique: int
    histogram: dict[int, int]
    best: tuple[ExpansionSeedRecord, ...]


def score_state(state: PairPermutations) -> int:
    score, _ = score_and_badness(
        np.asarray(state.P, dtype=np.int32),
        np.asarray(state.Q, dtype=np.int32),
    )
    return int(score)


def generate_projection_seeds(
    source: PairPermutations,
    *,
    keep: int = 100,
    include_d4: bool = True,
    include_decompositions: bool = True,
) -> SeedExperiment:
    source_points = source.points()
    variants = d4_variants(source_points, source.n) if include_d4 else [source_points]
    histogram: Counter[int] = Counter()
    seen: set[tuple[tuple[int, int], ...]] = set()
    # max heap represented by negative ranking values; smallest quality is removed manually.
    retained: list[SeedRecord] = []
    evaluated = 0

    for variant_index, points in enumerate(variants):
        decompositions = matching_decompositions(points, source.n) if include_decompositions else [matching_decompositions(points, source.n)[0]]
        for decomposition_index, variant_state in enumerate(decompositions):
            for removed_row in range(source.n):
                for removed_column in range(source.n):
                    evaluated += 1
                    candidate = project_state(variant_state, removed_row, removed_column)
                    signature = tuple(candidate.points())
                    if signature in seen:
                        continue
                    seen.add(signature)
                    score = score_state(candidate)
                    histogram[score] += 1
                    record = SeedRecord(score, removed_row, removed_column, variant_index, decomposition_index, candidate)
                    if len(retained) < keep:
                        retained.append(record)
                    else:
                        worst_index = max(range(len(retained)), key=lambda index: retained[index].score)
                        if score < retained[worst_index].score:
                            retained[worst_index] = record

    retained.sort(key=lambda record: (record.score, record.variant, record.decomposition, record.removed_row, record.removed_column))
    return SeedExperiment(evaluated, len(seen), dict(sorted(histogram.items())), tuple(retained))


def generate_expansion_seeds(
    source: PairPermutations,
    *,
    keep: int = 100,
    include_d4: bool = True,
    include_decompositions: bool = True,
) -> ExpansionSeedExperiment:
    """Expand a valid n-state to n+1 while preserving the 2-factor constraints."""
    source_points = source.points()
    variants = d4_variants(source_points, source.n) if include_d4 else [source_points]
    histogram: Counter[int] = Counter()
    seen: set[tuple[tuple[int, int], ...]] = set()
    retained: list[ExpansionSeedRecord] = []
    evaluated = 0

    def retain(record: ExpansionSeedRecord) -> None:
        if len(retained) < keep:
            retained.append(record)
            return
        worst_index = max(range(len(retained)), key=lambda index: retained[index].score)
        if record.score < retained[worst_index].score:
            retained[worst_index] = record

    for variant_index, points in enumerate(variants):
        all_decompositions = matching_decompositions(points, source.n)
        decompositions = all_decompositions if include_decompositions else all_decompositions[:1]
        for decomposition_index, variant_state in enumerate(decompositions):
            choices: list[tuple[int | None, int | None]] = []
            choices.extend((None, row) for row in range(source.n))
            choices.extend((row, None) for row in range(source.n))
            choices.extend(
                (row_p, row_q)
                for row_p in range(source.n)
                for row_q in range(source.n)
                if row_p != row_q and variant_state.P[row_p] != variant_state.Q[row_q]
            )
            for break_p, break_q in choices:
                evaluated += 1
                candidate = expand_state(
                    variant_state,
                    source.n,
                    source.n,
                    break_p=break_p,
                    break_q=break_q,
                )
                signature = tuple(candidate.points())
                if signature in seen:
                    continue
                seen.add(signature)
                score = score_state(candidate)
                histogram[score] += 1
                retain(
                    ExpansionSeedRecord(
                        score,
                        break_p,
                        break_q,
                        variant_index,
                        decomposition_index,
                        candidate,
                    )
                )

    retained.sort(
        key=lambda record: (
            record.score,
            record.variant,
            record.decomposition,
            -1 if record.break_p is None else record.break_p,
            -1 if record.break_q is None else record.break_q,
        )
    )
    return ExpansionSeedExperiment(evaluated, len(seen), dict(sorted(histogram.items())), tuple(retained))
