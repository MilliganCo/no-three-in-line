from no_three_75.geometry import canonical_line, determinant


def test_determinant_geometry() -> None:
    assert determinant((0, 0), (3, 2), (6, 4)) == 0
    assert determinant((0, 0), (3, 2), (6, 5)) != 0
    assert determinant((2, 0), (2, 4), (2, 9)) == 0
    assert determinant((0, 7), (4, 7), (9, 7)) == 0


def test_line_normalization_is_order_independent() -> None:
    expected = canonical_line((1, 2), (4, 8))
    assert canonical_line((4, 8), (1, 2)) == expected
    assert canonical_line((7, 14), (1, 2)) == expected
