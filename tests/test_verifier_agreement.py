from itertools import combinations
from random import Random

from no_three_75.verifier_a import find_collinear_triple as determinant_witness
from no_three_75.verifier_b import find_collinear_triple as slope_witness


def test_independent_verifiers_agree_on_random_small_sets() -> None:
    rng = Random(20260904)
    for _ in range(3000):
        n = rng.randint(2, 9)
        grid = [(row, column) for row in range(n) for column in range(n)]
        size = rng.randint(0, min(14, len(grid)))
        points = rng.sample(grid, size)
        assert (determinant_witness(points) is None) == (slope_witness(points) is None)


def test_opposite_rays_are_treated_as_same_line() -> None:
    points = [(0, 0), (2, 2), (5, 5)]
    assert determinant_witness(points) is not None
    assert slope_witness(points) is not None
