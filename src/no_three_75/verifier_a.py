from __future__ import annotations

from itertools import combinations

from .io import Point


def collinear(a: Point, b: Point, c: Point) -> bool:
    return (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0])


def find_collinear_triple(points: list[Point]) -> tuple[Point, Point, Point] | None:
    for a, b, c in combinations(points, 3):
        if collinear(a, b, c):
            return a, b, c
    return None


def verify(points: list[Point], n: int) -> tuple[bool, str]:
    if len(points) != 2 * n:
        return False, f"expected {2*n} points, got {len(points)}"
    if len(set(points)) != len(points):
        return False, "duplicate point"
    if any(not (0 <= row < n and 0 <= column < n) for row, column in points):
        return False, "point outside grid"
    rows = [0] * n
    columns = [0] * n
    for row, column in points:
        rows[row] += 1
        columns[column] += 1
    if any(value != 2 for value in rows):
        return False, "row degree is not exactly two"
    if any(value != 2 for value in columns):
        return False, "column degree is not exactly two"
    witness = find_collinear_triple(points)
    if witness is not None:
        return False, f"collinear triple: {witness}"
    return True, f"checked {len(points)*(len(points)-1)*(len(points)-2)//6} integer determinants"
