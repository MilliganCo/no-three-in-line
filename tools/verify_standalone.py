#!/usr/bin/env python3
"""Dependency-free exact verifier suitable for shipping with a witness."""

import argparse
import json
from collections import Counter
from itertools import combinations


def verify(path, n):
    if type(n) is not int or n < 1:
        return False, "n must be a positive integer"
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        if "n" in payload and (type(payload["n"]) is not int or payload["n"] < 1):
            return False, "JSON n must be a positive integer"
        points_data = payload.get("points")
    else:
        points_data = payload
    if not isinstance(points_data, list):
        return False, "JSON must contain a points list"
    if any(not isinstance(point, list) or len(point) != 2
           or any(type(coordinate) is not int for coordinate in point)
           for point in points_data):
        return False, "every point must contain exactly two integers"
    points = [(point[0], point[1]) for point in points_data]
    if len(points) != 2 * n:
        return False, f"expected {2 * n} points, found {len(points)}"
    if len(set(points)) != len(points):
        return False, "duplicate point"
    if any(not (0 <= row < n and 0 <= column < n) for row, column in points):
        return False, "point outside the grid"
    if set(Counter(row for row, _ in points).values()) != {2}:
        return False, "row degrees are not all two"
    if set(Counter(column for _, column in points).values()) != {2}:
        return False, "column degrees are not all two"
    checked = 0
    for a, b, c in combinations(points, 3):
        checked += 1
        if (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0]):
            return False, f"collinear triple: {a}, {b}, {c}"
    return True, f"checked {checked} exact integer determinants"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--n", type=int, default=75)
    args = parser.parse_args()
    try:
        valid, detail = verify(args.candidate, args.n)
    except (OSError, json.JSONDecodeError) as error:
        valid, detail = False, str(error)
    print(f"{'VALID' if valid else 'INVALID'}: {detail}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
