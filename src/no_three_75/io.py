from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

Point = tuple[int, int]

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%&@?!()[]<>{}=*+|-/~^_:;,."
SYMMETRY_PREFIXES = ".:/-ocx+*"
SYMMETRY_NAMES = {
    ".": "iden",
    ":": "rot2",
    "/": "dia1",
    "-": "ort1",
    "o": "rot4",
    "c": "rct4",
    "x": "dia2",
    "+": "ort2",
    "*": "full",
}



def _require_grid_n(value: object, label: str = "n") -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _json_points(value: object) -> list[Point]:
    if not isinstance(value, list):
        raise ValueError("JSON points must be a list")
    points: list[Point] = []
    for index, pair in enumerate(value):
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(coordinate) is not int for coordinate in pair)):
            raise ValueError(f"point {index} must be a list of exactly two integers")
        points.append((pair[0], pair[1]))
    return points


def decode_compact(code: str, n: int | None = None) -> tuple[int, str, list[Point]]:
    if n is not None:
        n = _require_grid_n(n)
    code = code.strip()
    if not code or code[0] not in SYMMETRY_PREFIXES:
        raise ValueError("compact code has no recognized symmetry prefix")
    payload = code[1:]
    if n is None:
        if len(payload) % 2:
            raise ValueError("compact payload length is not even")
        n = len(payload) // 2
    n = _require_grid_n(n)
    if len(payload) != 2 * n:
        raise ValueError(f"compact payload has {len(payload)} columns, expected {2*n}")
    index = {ch: value for value, ch in enumerate(ALPHABET)}
    points: list[Point] = []
    for offset, ch in enumerate(payload):
        if ch not in index:
            raise ValueError(f"unknown compact character {ch!r}")
        column = index[ch]
        if column >= n:
            raise ValueError(f"column {column} outside n={n}")
        points.append((offset // 2, column))
    return n, SYMMETRY_NAMES[code[0]], points


def encode_compact(points: Iterable[Point], n: int, symmetry: str = "iden") -> str:
    reverse_symmetry = {name: prefix for prefix, name in SYMMETRY_NAMES.items()}
    prefix = reverse_symmetry.get(symmetry, symmetry if symmetry in SYMMETRY_PREFIXES else None)
    if prefix is None:
        raise ValueError(f"unknown symmetry {symmetry!r}")
    rows: list[list[int]] = [[] for _ in range(n)]
    for row, column in points:
        if not (0 <= row < n and 0 <= column < n):
            raise ValueError(f"point {(row, column)} outside n={n}")
        rows[row].append(column)
    if any(len(row) != 2 for row in rows):
        raise ValueError("compact format requires exactly two points in every row")
    return prefix + "".join(ALPHABET[column] for row in rows for column in sorted(row))


def load_candidate(path: str | Path, n: int | None = None) -> tuple[int, list[Point], dict]:
    if n is not None:
        n = _require_grid_n(n)
    candidate_path = Path(path)
    text = candidate_path.read_text(encoding="utf-8").strip()
    metadata: dict = {"source_path": str(candidate_path)}

    if candidate_path.suffix.lower() == ".json" or text.startswith("{") or text.startswith("["):
        payload = json.loads(text)
        if isinstance(payload, dict):
            file_n = _require_grid_n(payload["n"], "JSON n") if "n" in payload else None
            points_data = payload.get("points")
            if points_data is None and "P" in payload and "Q" in payload:
                if not isinstance(payload["P"], list) or not isinstance(payload["Q"], list):
                    raise ValueError("JSON P and Q must be lists")
                points_data = [[r, c] for r, c in enumerate(payload["P"])] + [
                    [r, c] for r, c in enumerate(payload["Q"])
                ]
            if points_data is None:
                raise ValueError("JSON candidate has neither points nor P/Q")
            metadata.update({key: value for key, value in payload.items() if key != "points"})
        elif isinstance(payload, list):
            points_data, file_n = payload, None
        else:
            raise ValueError("JSON candidate must be an object or a points list")
        points = _json_points(points_data)
        inferred_n = max((row for row, _ in points), default=-1) + 1
        resolved_n = n if n is not None else file_n if file_n is not None else inferred_n
        return _require_grid_n(resolved_n), points, metadata

    tokens = text.split()
    if len(tokens) == 1 and tokens[0][0] in SYMMETRY_PREFIXES:
        decoded_n, symmetry, points = decode_compact(tokens[0], n=n)
        metadata.update({"compact": tokens[0], "symmetry": symmetry})
        return decoded_n, points, metadata

    values = [int(value) for value in re.findall(r"-?\d+", text)]
    if len(values) % 2:
        raise ValueError("text candidate contains an odd number of integers")
    points = list(zip(values[::2], values[1::2]))
    inferred_n = max((row for row, _ in points), default=-1) + 1
    return _require_grid_n(n if n is not None else inferred_n), points, metadata


def write_candidate(path: str | Path, n: int, points: Iterable[Point], **metadata: object) -> None:
    ordered = sorted((int(row), int(column)) for row, column in points)
    payload = {"n": int(n), "points": [[row, column] for row, column in ordered], **metadata}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
