from __future__ import annotations

import hashlib
from collections import Counter
from itertools import combinations

import numpy as np

from .accelerated import evaluate_two_switches, score_and_badness
from .conflicts import analyze_points
from .geometry import d4_variants
from .io import Point
from .state import PairPermutations, points_to_permutations


def bipartite_cycle_lengths(state: PairPermutations) -> tuple[int, ...]:
    inverse_p = [0] * state.n
    for row, column in enumerate(state.P):
        inverse_p[column] = row
    successor = [inverse_p[state.Q[row]] for row in range(state.n)]
    seen: set[int] = set()
    lengths: list[int] = []
    for start in range(state.n):
        if start in seen:
            continue
        row = start
        rows = 0
        while row not in seen:
            seen.add(row)
            rows += 1
            row = successor[row]
        lengths.append(2 * rows)
    return tuple(sorted(lengths))


def canonical_d4_signature(points: list[Point], n: int) -> tuple[Point, ...]:
    return min(tuple(sorted(variant)) for variant in d4_variants(points, n))


def pair_slope_histogram(points: list[Point]) -> Counter[tuple[int, int]]:
    from math import gcd

    histogram: Counter[tuple[int, int]] = Counter()
    for left, right in combinations(points, 2):
        delta_row = right[0] - left[0]
        delta_column = right[1] - left[1]
        divisor = gcd(abs(delta_row), abs(delta_column))
        delta_row //= divisor
        delta_column //= divisor
        if delta_row < 0 or (delta_row == 0 and delta_column < 0):
            delta_row = -delta_row
            delta_column = -delta_column
        histogram[(delta_row, delta_column)] += 1
    return histogram


def best_two_switch_score(state: PairPermutations) -> tuple[int, int]:
    p = np.asarray(state.P, dtype=np.int32)
    q = np.asarray(state.Q, dtype=np.int32)
    score, badness = score_and_badness(p, q)
    best = 1 << 60
    count = 0
    for selected in range(2 * state.n):
        scores = evaluate_two_switches(p, q, selected, score, badness)
        for partner in range(selected + 1, 2 * state.n):
            value = int(scores[partner])
            if value >= (1 << 29):
                continue
            if value < best:
                best = value
                count = 1
            elif value == best:
                count += 1
    return int(best), count


def structure_record(n: int, points: list[Point], *, include_neighborhood: bool = False) -> dict[str, object]:
    state = points_to_permutations(points, n)
    metrics = analyze_points(points)
    variants = d4_variants(points, n)
    signature = canonical_d4_signature(points, n)
    slopes = pair_slope_histogram(points)
    bad_points = [
        {"point": list(point), "badness": int(metrics.badness[index])}
        for index, point in enumerate(points)
        if metrics.badness[index]
    ]
    bad_points.sort(key=lambda item: (-item["badness"], item["point"]))
    record: dict[str, object] = {
        "n": n,
        "points": len(points),
        "triples": metrics.triples,
        "violating_lines": metrics.violating_lines,
        "max_point_badness": metrics.max_point_badness,
        "conflict_rows": sorted({row for line in metrics.lines for row, _ in line.points}),
        "bad_points": bad_points,
        "conflict_lines": [
            {"key": list(line.key), "points": [list(point) for point in line.points], "triples": line.triples}
            for line in metrics.lines
        ],
        "bipartite_cycle_lengths": list(bipartite_cycle_lengths(state)),
        "cycle_count": len(bipartite_cycle_lengths(state)),
        "d4_orbit_size": len(variants),
        "d4_stabilizer_size": 8 // len(variants),
        "canonical_d4_sha256": hashlib.sha256(repr(signature).encode("ascii")).hexdigest(),
        "top_pair_slopes": [
            {"slope": list(slope), "pairs": count}
            for slope, count in sorted(slopes.items(), key=lambda item: (-item[1], item[0]))[:15]
        ],
    }
    if include_neighborhood:
        minimum, ties = best_two_switch_score(state)
        record["two_switch_minimum"] = minimum
        record["two_switch_minimum_count"] = ties
    return record
