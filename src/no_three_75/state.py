from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .io import Point


@dataclass(frozen=True)
class PairPermutations:
    P: tuple[int, ...]
    Q: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.P)

    def validate(self) -> None:
        n = self.n
        expected = set(range(n))
        if len(self.Q) != n:
            raise ValueError("P and Q have different lengths")
        if set(self.P) != expected:
            raise ValueError("P is not a permutation")
        if set(self.Q) != expected:
            raise ValueError("Q is not a permutation")
        collisions = [row for row in range(n) if self.P[row] == self.Q[row]]
        if collisions:
            raise ValueError(f"P/Q duplicate edges at rows {collisions}")

    def points(self) -> list[Point]:
        self.validate()
        return sorted([(row, self.P[row]) for row in range(self.n)] + [(row, self.Q[row]) for row in range(self.n)])

    def swapped(self, layer: int, row_a: int, row_b: int) -> "PairPermutations":
        if layer not in (0, 1):
            raise ValueError("layer must be 0 (P) or 1 (Q)")
        p = list(self.P)
        q = list(self.Q)
        target = p if layer == 0 else q
        target[row_a], target[row_b] = target[row_b], target[row_a]
        candidate = PairPermutations(tuple(p), tuple(q))
        candidate.validate()
        return candidate


def points_to_permutations(points: list[Point], n: int) -> PairPermutations:
    adjacency: list[list[int]] = [[] for _ in range(n)]
    if len(points) != 2 * n or len(set(points)) != len(points):
        raise ValueError("a 2-factor needs exactly 2n distinct points")
    column_degree = [0] * n
    for row, column in points:
        if not (0 <= row < n and 0 <= column < n):
            raise ValueError(f"point {(row, column)} outside grid")
        adjacency[row].append(column)
        column_degree[column] += 1
    if any(len(neighbors) != 2 for neighbors in adjacency) or any(value != 2 for value in column_degree):
        raise ValueError("points are not a row/column 2-factor")
    for neighbors in adjacency:
        neighbors.sort()

    matched_row_for_column = [-1] * n

    def augment(row: int, seen: list[bool]) -> bool:
        for column in adjacency[row]:
            if seen[column]:
                continue
            seen[column] = True
            previous = matched_row_for_column[column]
            if previous == -1 or augment(previous, seen):
                matched_row_for_column[column] = row
                return True
        return False

    for row in range(n):
        if not augment(row, [False] * n):
            raise ValueError("2-factor unexpectedly has no perfect matching")

    p = [-1] * n
    for column, row in enumerate(matched_row_for_column):
        p[row] = column
    q = [next(column for column in adjacency[row] if column != p[row]) for row in range(n)]
    state = PairPermutations(tuple(p), tuple(q))
    state.validate()
    return state


def random_state(n: int, rng: Random) -> PairPermutations:
    p = list(range(n))
    q = list(range(n))
    rng.shuffle(p)
    while True:
        rng.shuffle(q)
        if all(p[row] != q[row] for row in range(n)):
            return PairPermutations(tuple(p), tuple(q))


def delete_row_and_column(permutation: tuple[int, ...], removed_row: int, removed_column: int) -> tuple[int, ...]:
    n = len(permutation)
    if not (0 <= removed_row < n and 0 <= removed_column < n):
        raise ValueError("deletion index outside permutation")
    values = list(permutation)
    source_of_removed_column = values.index(removed_column)
    if source_of_removed_column != removed_row:
        values[source_of_removed_column] = values[removed_row]
    result = []
    for row, value in enumerate(values):
        if row == removed_row:
            continue
        if value == removed_column:
            raise AssertionError("removed codomain value survived reconnection")
        result.append(value - (value > removed_column))
    expected = set(range(n - 1))
    if set(result) != expected:
        raise AssertionError("projection did not produce a permutation")
    return tuple(result)


def insert_row_and_column(
    permutation: tuple[int, ...],
    inserted_row: int,
    inserted_column: int,
    *,
    break_row: int | None,
) -> tuple[int, ...]:
    """Lift a permutation to n+1 with a direct edge or one spliced old edge."""
    n = len(permutation)
    if set(permutation) != set(range(n)):
        raise ValueError("input is not a permutation")
    if not (0 <= inserted_row <= n and 0 <= inserted_column <= n):
        raise ValueError("insertion index outside expanded permutation")
    if break_row is not None and not (0 <= break_row < n):
        raise ValueError("break row outside source permutation")

    def shifted_row(row: int) -> int:
        return row + (row >= inserted_row)

    def shifted_column(column: int) -> int:
        return column + (column >= inserted_column)

    result = [-1] * (n + 1)
    for row, column in enumerate(permutation):
        result[shifted_row(row)] = shifted_column(column)
    if break_row is None:
        result[inserted_row] = inserted_column
    else:
        old_column = shifted_column(permutation[break_row])
        result[shifted_row(break_row)] = inserted_column
        result[inserted_row] = old_column
    if set(result) != set(range(n + 1)):
        raise AssertionError("expansion did not produce a permutation")
    return tuple(result)


def expand_state(
    state: PairPermutations,
    inserted_row: int,
    inserted_column: int,
    *,
    break_p: int | None,
    break_q: int | None,
) -> PairPermutations:
    expanded = PairPermutations(
        insert_row_and_column(
            state.P,
            inserted_row,
            inserted_column,
            break_row=break_p,
        ),
        insert_row_and_column(
            state.Q,
            inserted_row,
            inserted_column,
            break_row=break_q,
        ),
    )
    expanded.validate()
    return expanded


def repair_collisions(state: PairPermutations) -> PairPermutations:
    p = list(state.P)
    q = list(state.Q)
    n = len(p)
    guard = 0
    while True:
        collisions = [row for row in range(n) if p[row] == q[row]]
        if not collisions:
            repaired = PairPermutations(tuple(p), tuple(q))
            repaired.validate()
            return repaired
        row = collisions[0]
        partner = next(
            (
                other
                for other in range(n)
                if other != row and q[other] != p[row] and q[row] != p[other]
            ),
            None,
        )
        if partner is None:
            raise ValueError("could not repair a P/Q collision by transposition")
        q[row], q[partner] = q[partner], q[row]
        guard += 1
        if guard > 2 * n:
            raise AssertionError("collision repair failed to converge")


def project_state(state: PairPermutations, removed_row: int, removed_column: int) -> PairPermutations:
    projected = PairPermutations(
        delete_row_and_column(state.P, removed_row, removed_column),
        delete_row_and_column(state.Q, removed_row, removed_column),
    )
    return repair_collisions(projected)


def matching_decompositions(points: list[Point], n: int, *, max_components: int = 16) -> list[PairPermutations]:
    """Enumerate the two-colourings of the 2-factor, modulo a global P/Q swap."""
    base = points_to_permutations(points, n)
    inverse_p = [0] * n
    for row, column in enumerate(base.P):
        inverse_p[column] = row
    successor = [inverse_p[base.Q[row]] for row in range(n)]
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(n):
        if start in seen:
            continue
        component: list[int] = []
        row = start
        while row not in seen:
            seen.add(row)
            component.append(row)
            row = successor[row]
        components.append(component)
    if len(components) > max_components:
        raise ValueError(
            f"2-factor has {len(components)} components; exhaustive matching colourings would be too large"
        )

    states: list[PairPermutations] = []
    # Swapping every component only exchanges P and Q globally, so fix component 0.
    for mask in range(1 << max(0, len(components) - 1)):
        p = list(base.P)
        q = list(base.Q)
        for component_index in range(1, len(components)):
            if not (mask & (1 << (component_index - 1))):
                continue
            for row in components[component_index]:
                p[row], q[row] = q[row], p[row]
        state = PairPermutations(tuple(p), tuple(q))
        state.validate()
        states.append(state)
    return states
