from no_three_75.analysis import bipartite_cycle_lengths, canonical_d4_signature
from no_three_75.state import PairPermutations


def test_cycle_lengths_cover_every_bipartite_vertex() -> None:
    state = PairPermutations((0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4))
    assert bipartite_cycle_lengths(state) == (4, 4, 4)
    assert sum(bipartite_cycle_lengths(state)) == 2 * state.n


def test_d4_canonical_signature_is_invariant() -> None:
    state = PairPermutations((0, 1, 2, 3), (1, 2, 3, 0))
    points = state.points()
    signature = canonical_d4_signature(points, 4)
    assert len(signature) == 8
