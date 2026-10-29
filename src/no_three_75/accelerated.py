from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised on dependency-minimal installs
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


@njit(cache=True)
def score_and_badness(p: np.ndarray, q: np.ndarray) -> tuple[int, np.ndarray]:
    n = len(p)
    count = 2 * n
    rows = np.empty(count, dtype=np.int32)
    columns = np.empty(count, dtype=np.int32)
    for row in range(n):
        rows[2 * row] = row
        columns[2 * row] = p[row]
        rows[2 * row + 1] = row
        columns[2 * row + 1] = q[row]
    badness = np.zeros(count, dtype=np.int64)
    score = 0
    for i in range(count - 2):
        for j in range(i + 1, count - 1):
            dx1 = rows[j] - rows[i]
            dy1 = columns[j] - columns[i]
            for k in range(j + 1, count):
                if dx1 * (columns[k] - columns[i]) == dy1 * (rows[k] - rows[i]):
                    score += 1
                    badness[i] += 1
                    badness[j] += 1
                    badness[k] += 1
    return score, badness


@njit(cache=True)
def evaluate_transpositions(
    p: np.ndarray,
    q: np.ndarray,
    layer: int,
    selected_row: int,
    current_score: int,
    badness: np.ndarray,
) -> np.ndarray:
    n = len(p)
    count = 2 * n
    rows = np.empty(count, dtype=np.int32)
    columns = np.empty(count, dtype=np.int32)
    for row in range(n):
        rows[2 * row] = row
        columns[2 * row] = p[row]
        rows[2 * row + 1] = row
        columns[2 * row + 1] = q[row]
    scores = np.full(n, np.int64(1 << 60), dtype=np.int64)
    u = 2 * selected_row + layer

    for partner_row in range(n):
        if partner_row == selected_row:
            continue
        if layer == 0:
            if p[partner_row] == q[selected_row] or p[selected_row] == q[partner_row]:
                continue
        else:
            if q[partner_row] == p[selected_row] or q[selected_row] == p[partner_row]:
                continue
        v = 2 * partner_row + layer

        joint_old = 0
        for other in range(count):
            if other == u or other == v:
                continue
            if (rows[v] - rows[u]) * (columns[other] - columns[u]) == (columns[v] - columns[u]) * (rows[other] - rows[u]):
                joint_old += 1
        old_affected = badness[u] + badness[v] - joint_old

        new_u_row = rows[u]
        new_u_column = columns[v]
        new_v_row = rows[v]
        new_v_column = columns[u]
        new_affected = 0

        # Every triple containing u. Triples containing both u and v occur here once.
        for i in range(count - 1):
            if i == u:
                continue
            if i == v:
                i_row = new_v_row
                i_column = new_v_column
            else:
                i_row = rows[i]
                i_column = columns[i]
            for j in range(i + 1, count):
                if j == u:
                    continue
                if j == v:
                    j_row = new_v_row
                    j_column = new_v_column
                else:
                    j_row = rows[j]
                    j_column = columns[j]
                if (i_row - new_u_row) * (j_column - new_u_column) == (i_column - new_u_column) * (j_row - new_u_row):
                    new_affected += 1

        # Remaining affected triples contain v but not u.
        for i in range(count - 1):
            if i == u or i == v:
                continue
            for j in range(i + 1, count):
                if j == u or j == v:
                    continue
                if (rows[i] - new_v_row) * (columns[j] - new_v_column) == (columns[i] - new_v_column) * (rows[j] - new_v_row):
                    new_affected += 1

        scores[partner_row] = current_score - old_affected + new_affected
    return scores


@njit(cache=True)
def evaluate_two_switches(
    p: np.ndarray,
    q: np.ndarray,
    selected_index: int,
    current_score: int,
    badness: np.ndarray,
) -> np.ndarray:
    """Score every degree-preserving edge 2-switch involving one selected point."""
    n = len(p)
    count = 2 * n
    rows = np.empty(count, dtype=np.int32)
    columns = np.empty(count, dtype=np.int32)
    for row in range(n):
        rows[2 * row] = row
        columns[2 * row] = p[row]
        rows[2 * row + 1] = row
        columns[2 * row + 1] = q[row]
    scores = np.full(count, np.int64(1 << 60), dtype=np.int64)
    u = selected_index
    selected_row = rows[u]
    selected_other = 2 * selected_row + (1 - (u & 1))

    for v in range(count):
        if v == u or rows[v] == selected_row:
            continue
        partner_row = rows[v]
        partner_other = 2 * partner_row + (1 - (v & 1))
        if columns[v] == columns[selected_other] or columns[u] == columns[partner_other]:
            continue

        joint_old = 0
        for other in range(count):
            if other == u or other == v:
                continue
            if (rows[v] - rows[u]) * (columns[other] - columns[u]) == (columns[v] - columns[u]) * (rows[other] - rows[u]):
                joint_old += 1
        old_affected = badness[u] + badness[v] - joint_old

        new_u_row = rows[u]
        new_u_column = columns[v]
        new_v_row = rows[v]
        new_v_column = columns[u]
        new_affected = 0
        for i in range(count - 1):
            if i == u:
                continue
            if i == v:
                i_row = new_v_row
                i_column = new_v_column
            else:
                i_row = rows[i]
                i_column = columns[i]
            for j in range(i + 1, count):
                if j == u:
                    continue
                if j == v:
                    j_row = new_v_row
                    j_column = new_v_column
                else:
                    j_row = rows[j]
                    j_column = columns[j]
                if (i_row - new_u_row) * (j_column - new_u_column) == (i_column - new_u_column) * (j_row - new_u_row):
                    new_affected += 1
        for i in range(count - 1):
            if i == u or i == v:
                continue
            for j in range(i + 1, count):
                if j == u or j == v:
                    continue
                if (rows[i] - new_v_row) * (columns[j] - new_v_column) == (columns[i] - new_v_column) * (rows[j] - new_v_row):
                    new_affected += 1
        scores[v] = current_score - old_affected + new_affected
    return scores


@njit(cache=True)
def evaluate_three_cycles(
    p: np.ndarray,
    q: np.ndarray,
    candidates: np.ndarray,
    current_score: int,
    badness: np.ndarray,
) -> np.ndarray:
    """Score legal candidates encoded as (u, v, w, orientation)."""
    n = len(p)
    count = 2 * n
    rows = np.empty(count, dtype=np.int32)
    columns = np.empty(count, dtype=np.int32)
    for row in range(n):
        rows[2 * row] = row
        columns[2 * row] = p[row]
        rows[2 * row + 1] = row
        columns[2 * row + 1] = q[row]
    scores = np.full(len(candidates), np.int64(1 << 60), dtype=np.int64)

    for candidate_index in range(len(candidates)):
        u = int(candidates[candidate_index, 0])
        v = int(candidates[candidate_index, 1])
        w = int(candidates[candidate_index, 2])
        orientation = int(candidates[candidate_index, 3])

        joint_uv = 0
        joint_uw = 0
        joint_vw = 0
        for other in range(count):
            if other != u and other != v:
                if (rows[v] - rows[u]) * (columns[other] - columns[u]) == (columns[v] - columns[u]) * (rows[other] - rows[u]):
                    joint_uv += 1
            if other != u and other != w:
                if (rows[w] - rows[u]) * (columns[other] - columns[u]) == (columns[w] - columns[u]) * (rows[other] - rows[u]):
                    joint_uw += 1
            if other != v and other != w:
                if (rows[w] - rows[v]) * (columns[other] - columns[v]) == (columns[w] - columns[v]) * (rows[other] - rows[v]):
                    joint_vw += 1
        triple_uvw = 1 if (rows[v] - rows[u]) * (columns[w] - columns[u]) == (columns[v] - columns[u]) * (rows[w] - rows[u]) else 0
        old_affected = badness[u] + badness[v] + badness[w] - joint_uv - joint_uw - joint_vw + triple_uvw

        changed_columns = columns.copy()
        if orientation == 0:
            changed_columns[u] = columns[v]
            changed_columns[v] = columns[w]
            changed_columns[w] = columns[u]
        else:
            changed_columns[u] = columns[w]
            changed_columns[w] = columns[v]
            changed_columns[v] = columns[u]

        new_affected = 0
        for i in range(count - 1):
            if i == u:
                continue
            for j in range(i + 1, count):
                if j == u:
                    continue
                if (rows[i] - rows[u]) * (changed_columns[j] - changed_columns[u]) == (changed_columns[i] - changed_columns[u]) * (rows[j] - rows[u]):
                    new_affected += 1
        for i in range(count - 1):
            if i == u or i == v:
                continue
            for j in range(i + 1, count):
                if j == u or j == v:
                    continue
                if (rows[i] - rows[v]) * (changed_columns[j] - changed_columns[v]) == (changed_columns[i] - changed_columns[v]) * (rows[j] - rows[v]):
                    new_affected += 1
        for i in range(count - 1):
            if i == u or i == v or i == w:
                continue
            for j in range(i + 1, count):
                if j == u or j == v or j == w:
                    continue
                if (rows[i] - rows[w]) * (changed_columns[j] - changed_columns[w]) == (changed_columns[i] - changed_columns[w]) * (rows[j] - rows[w]):
                    new_affected += 1
        scores[candidate_index] = current_score - old_affected + new_affected
    return scores
