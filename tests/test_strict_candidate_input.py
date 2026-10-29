import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from no_three_75.io import decode_compact, load_candidate
from no_three_75.verifier_a import verify as verify_a
from no_three_75.verifier_b import verify as verify_b

ROOT = Path(__file__).resolve().parents[1]
SQUARE = [[0, 0], [0, 1], [1, 0], [1, 1]]
spec = importlib.util.spec_from_file_location('standalone', ROOT/'tools/verify_standalone.py')
standalone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standalone)  # stdlib definitions; __main__ does not run


def save(tmp_path, payload):
    path = tmp_path/'synthetic.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


@pytest.mark.parametrize('points', [
    [[bool(r), bool(c)] for r, c in SQUARE],
    [[float(r), float(c)] for r, c in SQUARE],
    [[r + .25, c + .75] for r, c in SQUARE],
    [[str(r), str(c)] for r, c in SQUARE],
    ['00', '01', '10', '11'],
    [[0, 0, 99], [0, 1], [1, 0], [1, 1]],
    [[0], [0, 1], [1, 0], [1, 1]],
    [None, [0, 1], [1, 0], [1, 1]],
])
def test_json_coordinates_are_never_coerced(tmp_path, points):
    path = save(tmp_path, {'n': 2, 'points': points})
    with pytest.raises(ValueError):
        load_candidate(path)
    valid, detail = standalone.verify(path, 2)
    assert not valid and 'two integers' in detail


@pytest.mark.parametrize('points', [None, True, 3, '00', {'0': [0, 0]}])
def test_points_container_is_a_list(tmp_path, points):
    path = save(tmp_path, {'n': 2, 'points': points})
    with pytest.raises(ValueError):
        load_candidate(path)
    assert standalone.verify(path, 2)[0] is False


@pytest.mark.parametrize('bad_n', [True, False, 2.0, 2.5, '2', None, 0, -1])
def test_declared_json_n_is_strict_even_with_override(tmp_path, bad_n):
    path = save(tmp_path, {'n': bad_n, 'points': SQUARE})
    for explicit in (None, 2):
        with pytest.raises(ValueError, match='positive integer'):
            load_candidate(path, n=explicit)
    assert standalone.verify(path, 2)[0] is False


@pytest.mark.parametrize('bad_n', [True, False, 2.0, 2.5, '2', 0, -1])
def test_api_n_is_strict(tmp_path, bad_n):
    path = save(tmp_path, {'n': 2, 'points': SQUARE})
    with pytest.raises(ValueError, match='positive integer'):
        load_candidate(path, n=bad_n)
    with pytest.raises(ValueError, match='positive integer'):
        decode_compact('.0101', n=bad_n)
    valid, detail = standalone.verify(path, bad_n)
    assert not valid and 'positive integer' in detail


@pytest.mark.parametrize('payload', [SQUARE, {'points': SQUARE},
    {'n': 2, 'points': SQUARE, 'label': 'synthetic'},
    {'n': 2, 'P': [0, 1], 'Q': [1, 0]},
])
def test_valid_integers_remain_zero_based_and_unchanged(tmp_path, payload):
    n, points, metadata = load_candidate(save(tmp_path, payload))
    assert n == 2 and sorted(points) == [tuple(p) for p in SQUARE]
    assert all(type(x) is int for point in points for x in point)
    assert verify_a(points, n)[0] and verify_b(points, n)[0]
    if isinstance(payload, dict) and 'label' in payload:
        assert metadata['label'] == payload['label']


def test_override_and_compact_text_contracts_unchanged(tmp_path):
    path = save(tmp_path, {'n': 2, 'points': SQUARE})
    assert load_candidate(path, n=3)[0] == 3  # valid explicit override preserved
    assert decode_compact('.0101')[2] == [tuple(p) for p in SQUARE]
    text = tmp_path/'synthetic.txt'
    text.write_text('0 0\n0 1\n1 0\n1 1\n', encoding='utf-8')
    assert load_candidate(text)[:2] == (2, [tuple(p) for p in SQUARE])
    one_based = [[r + 1, c + 1] for r, c in SQUARE]
    n, points, _ = load_candidate(save(tmp_path, {'n': 2, 'points': one_based}))
    assert points == [tuple(p) for p in one_based]  # never auto-shift
    assert not verify_a(points, n)[0] and not verify_b(points, n)[0]


@pytest.mark.parametrize('bad', [True, 0.0, '0', None, [0, 1]])
def test_pq_coordinates_are_strict(tmp_path, bad):
    payload = {'n': 2, 'P': [bad, 1], 'Q': [1, 0]}
    with pytest.raises(ValueError):
        load_candidate(save(tmp_path, payload))


@pytest.mark.parametrize('bad', ['01', {}, None, True])
def test_pq_containers_are_lists(tmp_path, bad):
    with pytest.raises(ValueError):
        load_candidate(save(tmp_path, {'n': 2, 'P': bad, 'Q': [1, 0]}))


@pytest.mark.parametrize('script', ['verify_solution.py', 'verify_solution_alt.py',
                                   'verify_standalone.py'])
def test_public_verifier_clis_accept_integer_square_and_reject_float_alias(tmp_path, script):
    for points, expected_code, prefix in [
        (SQUARE, 0, 'VALID'),
        ([[r + .25, c + .75] for r, c in SQUARE], 1, 'INVALID'),
    ]:
        path = save(tmp_path, {'n': 2, 'points': points})
        result = subprocess.run([sys.executable, str(ROOT/'tools'/script),
                                 str(path), '--n', '2'],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == expected_code
        assert result.stdout.startswith(prefix)
