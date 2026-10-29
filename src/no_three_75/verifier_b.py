from __future__ import annotations

from math import gcd

from .io import Point


def primitive_undirected_direction(a: Point, b: Point) -> tuple[int, int]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    divisor = gcd(abs(dx), abs(dy))
    if divisor == 0:
        raise ValueError("duplicate point has no direction")
    dx //= divisor
    dy //= divisor
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return dx, dy


def find_collinear_triple(points: list[Point]) -> tuple[Point, Point, Point] | None:
    for anchor_index, anchor in enumerate(points):
        seen: dict[tuple[int, int], Point] = {}
        for other_index, other in enumerate(points):
            if other_index == anchor_index:
                continue
            direction = primitive_undirected_direction(anchor, other)
            previous = seen.get(direction)
            if previous is not None:
                return anchor, previous, other
            seen[direction] = other
    return None


def verify(points: list[Point], n: int) -> tuple[bool, str]:
    if len(points) != n + n:
        return False, f"cardinality mismatch: {len(points)} versus {2*n}"
    unique = set(points)
    if len(unique) != len(points):
        return False, "coordinates are not unique"
    row_counts = {row: 0 for row in range(n)}
    column_counts = {column: 0 for column in range(n)}
    for point in points:
        row, column = point
        if row not in row_counts or column not in column_counts:
            return False, f"out-of-range coordinate {point}"
        row_counts[row] += 1
        column_counts[column] += 1
    if set(row_counts.values()) != {2}:
        return False, "not every row occurs twice"
    if set(column_counts.values()) != {2}:
        return False, "not every column occurs twice"
    witness = find_collinear_triple(points)
    if witness:
        return False, f"repeated normalized direction at {witness[0]}: {witness}"
    return True, f"checked normalized directions around {len(points)} anchors"
