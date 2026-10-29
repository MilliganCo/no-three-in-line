from __future__ import annotations

from math import gcd

from .io import Point

LineKey = tuple[int, int, int]


def determinant(a: Point, b: Point, c: Point) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def canonical_line(a: Point, b: Point) -> LineKey:
    if a == b:
        raise ValueError("two distinct points are required")
    coefficient_a = b[1] - a[1]
    coefficient_b = a[0] - b[0]
    coefficient_c = -(coefficient_a * a[0] + coefficient_b * a[1])
    divisor = gcd(gcd(abs(coefficient_a), abs(coefficient_b)), abs(coefficient_c))
    coefficient_a //= divisor
    coefficient_b //= divisor
    coefficient_c //= divisor
    if coefficient_a < 0 or (coefficient_a == 0 and coefficient_b < 0):
        coefficient_a = -coefficient_a
        coefficient_b = -coefficient_b
        coefficient_c = -coefficient_c
    return coefficient_a, coefficient_b, coefficient_c


def primitive_slope(line: LineKey) -> tuple[int, int]:
    coefficient_a, coefficient_b, _ = line
    dx, dy = coefficient_b, -coefficient_a
    divisor = gcd(abs(dx), abs(dy))
    dx //= divisor
    dy //= divisor
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return dx, dy


def d4_variants(points: list[Point], n: int) -> list[list[Point]]:
    def transforms(point: Point) -> tuple[Point, ...]:
        row, column = point
        high = n - 1
        return (
            (row, column),
            (column, high - row),
            (high - row, high - column),
            (high - column, row),
            (row, high - column),
            (high - row, column),
            (column, row),
            (high - column, high - row),
        )

    buckets: list[list[Point]] = [[] for _ in range(8)]
    for point in points:
        for index, transformed in enumerate(transforms(point)):
            buckets[index].append(transformed)
    unique: dict[tuple[Point, ...], list[Point]] = {}
    for variant in buckets:
        ordered = sorted(variant)
        unique.setdefault(tuple(ordered), ordered)
    return list(unique.values())
