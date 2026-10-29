from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import comb

from .geometry import LineKey, canonical_line, determinant, primitive_slope
from .io import Point
from .state import PairPermutations


@dataclass(frozen=True)
class ViolatingLine:
    key: LineKey
    points: tuple[Point, ...]
    triples: int


@dataclass(frozen=True)
class ConflictMetrics:
    triples: int
    violating_lines: int
    max_line_occupancy: int
    max_point_badness: int
    weighted_score: int
    badness: tuple[int, ...]
    lines: tuple[ViolatingLine, ...]
    slope_histogram: dict[tuple[int, int], int]

    @property
    def lexicographic(self) -> tuple[int, int, int, int]:
        return self.triples, self.violating_lines, self.max_point_badness, self.weighted_score


def analyze_points(points: list[Point], line_weights: dict[LineKey, int] | None = None) -> ConflictMetrics:
    point_sets: dict[LineKey, set[int]] = {}
    for left, right in combinations(range(len(points)), 2):
        key = canonical_line(points[left], points[right])
        members = point_sets.setdefault(key, set())
        members.add(left)
        members.add(right)

    badness = [0] * len(points)
    lines: list[ViolatingLine] = []
    slope_histogram: Counter[tuple[int, int]] = Counter()
    triples = 0
    weighted_score = 0
    for key, indices in point_sets.items():
        occupancy = len(indices)
        if occupancy < 3:
            continue
        line_triples = comb(occupancy, 3)
        triples += line_triples
        weighted_score += (line_weights.get(key, 1) if line_weights else 1) * line_triples
        per_point = comb(occupancy - 1, 2)
        for index in indices:
            badness[index] += per_point
        line_points = tuple(sorted(points[index] for index in indices))
        lines.append(ViolatingLine(key, line_points, line_triples))
        slope_histogram[primitive_slope(key)] += line_triples
    lines.sort(key=lambda line: (-line.triples, line.key))
    return ConflictMetrics(
        triples=triples,
        violating_lines=len(lines),
        max_line_occupancy=max((len(line.points) for line in lines), default=2),
        max_point_badness=max(badness, default=0),
        weighted_score=weighted_score,
        badness=tuple(badness),
        lines=tuple(lines),
        slope_histogram=dict(slope_histogram),
    )


def full_triple_score(points: list[Point]) -> int:
    return sum(1 for a, b, c in combinations(points, 3) if determinant(a, b, c) == 0)


def reference_swap_delta(state: PairPermutations, layer: int, row_a: int, row_b: int) -> int:
    old_points = state.points()
    candidate = state.swapped(layer, row_a, row_b)
    new_points = candidate.points()
    return full_triple_score(new_points) - full_triple_score(old_points)
