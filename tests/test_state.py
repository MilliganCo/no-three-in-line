from random import Random

from no_three_75.state import (
    PairPermutations,
    expand_state,
    insert_row_and_column,
    matching_decompositions,
    project_state,
    random_state,
)


def test_random_state_invariants() -> None:
    state = random_state(75, Random(123))
    state.validate()
    assert len(state.points()) == 150


def test_projection_invariants_for_every_small_deletion() -> None:
    state = random_state(12, Random(5))
    for row in range(12):
        for column in range(12):
            projected = project_state(state, row, column)
            projected.validate()
            assert projected.n == 11


def test_legal_swap_preserves_invariants() -> None:
    state = PairPermutations((0, 1, 2, 3), (1, 2, 3, 0))
    moved = state.swapped(0, 0, 2)
    moved.validate()


def test_matching_decompositions_preserve_the_same_points() -> None:
    state = PairPermutations((0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4))
    decompositions = matching_decompositions(state.points(), 6)
    assert len(decompositions) == 4
    assert all(candidate.points() == state.points() for candidate in decompositions)


def test_insert_permutation_direct_and_spliced() -> None:
    permutation = (2, 0, 3, 1)
    assert insert_row_and_column(permutation, 4, 4, break_row=None) == (2, 0, 3, 1, 4)
    assert insert_row_and_column(permutation, 4, 4, break_row=1) == (2, 4, 3, 1, 0)
    assert insert_row_and_column(permutation, 0, 2, break_row=3) == (1, 3, 0, 4, 2)


def test_expand_state_preserves_two_factor() -> None:
    state = PairPermutations((0, 1, 2, 3), (1, 2, 3, 0))
    expanded = expand_state(state, 4, 4, break_p=None, break_q=2)
    expanded.validate()
    assert len(expanded.points()) == 10
