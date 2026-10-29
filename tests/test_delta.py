from random import Random

import numpy as np

from no_three_75.accelerated import evaluate_three_cycles, evaluate_transpositions, evaluate_two_switches, score_and_badness
from no_three_75.conflicts import full_triple_score
from no_three_75.state import random_state


def test_incremental_transposition_scores_match_full_recompute() -> None:
    rng = Random(9817)
    comparisons = 0
    for n in range(4, 13):
        for _ in range(8):
            state = random_state(n, rng)
            p = np.asarray(state.P, dtype=np.int32)
            q = np.asarray(state.Q, dtype=np.int32)
            score, badness = score_and_badness(p, q)
            assert int(score) == full_triple_score(state.points())
            for _ in range(4):
                layer = rng.randrange(2)
                row = rng.randrange(n)
                candidates = evaluate_transpositions(p, q, layer, row, score, badness)
                for partner in range(n):
                    if candidates[partner] >= (1 << 59):
                        continue
                    moved = state.swapped(layer, row, partner)
                    assert int(candidates[partner]) == full_triple_score(moved.points())
                    comparisons += 1
    assert comparisons >= 1000


def test_general_two_switch_scores_match_full_recompute() -> None:
    from no_three_75.state import points_to_permutations

    rng = Random(443)
    comparisons = 0
    for n in range(4, 11):
        for _ in range(5):
            state = random_state(n, rng)
            p = np.asarray(state.P, dtype=np.int32)
            q = np.asarray(state.Q, dtype=np.int32)
            score, badness = score_and_badness(p, q)
            selected = rng.randrange(2 * n)
            candidates = evaluate_two_switches(p, q, selected, score, badness)
            row_a, layer_a = divmod(selected, 2)
            column_a = state.P[row_a] if layer_a == 0 else state.Q[row_a]
            for partner in range(2 * n):
                if candidates[partner] >= (1 << 59):
                    continue
                row_b, layer_b = divmod(partner, 2)
                column_b = state.P[row_b] if layer_b == 0 else state.Q[row_b]
                points = state.points()
                points.remove((row_a, column_a))
                points.remove((row_b, column_b))
                points.extend([(row_a, column_b), (row_b, column_a)])
                moved = points_to_permutations(points, n)
                assert int(candidates[partner]) == full_triple_score(moved.points())
                comparisons += 1
    assert comparisons >= 200


def test_three_cycle_scores_match_full_recompute() -> None:
    from no_three_75.search import apply_three_cycle, legal_three_cycle_candidates

    rng = Random(991)
    comparisons = 0
    for n in range(4, 10):
        state = random_state(n, rng)
        p = np.asarray(state.P, dtype=np.int32)
        q = np.asarray(state.Q, dtype=np.int32)
        score, badness = score_and_badness(p, q)
        selected = rng.randrange(2 * n)
        candidates = legal_three_cycle_candidates(p, q, selected)
        if len(candidates) == 0:
            continue
        sample = candidates[: min(30, len(candidates))]
        scores = evaluate_three_cycles(p, q, sample, score, badness)
        for candidate, candidate_score in zip(sample, scores):
            moved = apply_three_cycle(p, q, candidate)
            assert int(candidate_score) == full_triple_score(moved.points())
            comparisons += 1
    assert comparisons >= 50
