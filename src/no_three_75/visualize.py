from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .conflicts import ConflictMetrics, analyze_points
from .io import Point


def _history_series(path: str | Path | None) -> tuple[list[float], list[int]]:
    if path is None:
        return [], []
    elapsed: list[float] = []
    scores: list[int] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        value = record.get("best_score", record.get("triples"))
        if value is None:
            continue
        elapsed.append(float(record.get("elapsed_seconds", record.get("round", len(elapsed) + 1))))
        scores.append(int(value))
    return elapsed, scores


def render_candidate(
    n: int,
    points: list[Point],
    output: str | Path,
    *,
    title: str | None = None,
    history: str | Path | None = None,
) -> ConflictMetrics:
    metrics = analyze_points(points)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    badness = np.asarray(metrics.badness, dtype=np.int64)
    rows = np.asarray([point[0] for point in points])
    columns = np.asarray([point[1] for point in points])

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(3.1, 1.35), height_ratios=(1, 1))
    board = figure.add_subplot(grid[:, 0])
    metrics_axis = figure.add_subplot(grid[0, 1])
    detail_axis = figure.add_subplot(grid[1, 1])

    board.set_facecolor("#f8fafc")
    board.scatter(columns[badness == 0], rows[badness == 0], s=24, c="#2563eb", alpha=0.78, linewidths=0)
    if np.any(badness):
        sizes = 48 + 24 * badness[badness > 0]
        board.scatter(
            columns[badness > 0],
            rows[badness > 0],
            s=sizes,
            c=badness[badness > 0],
            cmap="YlOrRd",
            vmin=1,
            vmax=max(2, int(badness.max())),
            edgecolors="#7f1d1d",
            linewidths=0.7,
            zorder=4,
        )
    for line in metrics.lines:
        ordered = sorted(line.points, key=lambda point: (point[1], point[0]))
        left, right = ordered[0], ordered[-1]
        board.plot(
            (left[1], right[1]),
            (left[0], right[0]),
            color="#dc2626",
            alpha=0.48,
            linewidth=1.35 + 0.25 * (len(line.points) - 3),
            zorder=2,
        )
    for index in np.argsort(-badness)[:8]:
        if badness[index] <= 1:
            continue
        board.annotate(
            f"{tuple(points[int(index)])} · {int(badness[index])}",
            (columns[index], rows[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#7f1d1d",
            zorder=5,
        )
    board.set_xlim(-1, n)
    board.set_ylim(n, -1)
    board.set_aspect("equal")
    board.set_xlabel("column")
    board.set_ylabel("row (0 at top)")
    board.set_xticks(range(0, n, 5))
    board.set_yticks(range(0, n, 5))
    board.grid(True, color="#cbd5e1", linewidth=0.45, alpha=0.7)
    board.set_title(title or f"No-three-in-line candidate: n={n}, E={metrics.triples}", loc="left", fontsize=15, fontweight="bold")

    metrics_axis.axis("off")
    lines = [
        ("points", len(points)),
        ("triple conflicts E", metrics.triples),
        ("violating lines", metrics.violating_lines),
        ("max line occupancy", metrics.max_line_occupancy),
        ("max point badness", metrics.max_point_badness),
    ]
    metrics_axis.text(0.0, 1.0, "Exact diagnostics", va="top", fontsize=13, fontweight="bold", color="#0f172a")
    for index, (label, value) in enumerate(lines):
        metrics_axis.text(0.02, 0.82 - index * 0.14, label, fontsize=10, color="#475569")
        metrics_axis.text(0.98, 0.82 - index * 0.14, str(value), ha="right", fontsize=12, fontweight="bold", color="#0f172a")
    metrics_axis.text(
        0.02,
        0.06,
        "Blue: conflict-free point\nOrange/red: point badness\nRed segments: violating lines",
        fontsize=9,
        color="#475569",
        va="bottom",
    )

    history_x, history_y = _history_series(history)
    if history_y:
        detail_axis.step(history_x, history_y, where="post", color="#7c3aed", linewidth=2)
        detail_axis.scatter(history_x, history_y, color="#7c3aed", s=18)
        detail_axis.set_title("Best score during search", loc="left", fontsize=12, fontweight="bold")
        detail_axis.set_xlabel("elapsed seconds / round")
        detail_axis.set_ylabel("E")
        detail_axis.set_ylim(bottom=0)
    else:
        slope_items = sorted(metrics.slope_histogram.items(), key=lambda item: (-item[1], item[0]))[:12]
        labels = [f"{dy}/{dx}" for (dx, dy), _ in slope_items]
        values = [value for _, value in slope_items]
        if values:
            positions = np.arange(len(values))
            detail_axis.barh(positions, values, color="#ef4444", alpha=0.78)
            detail_axis.set_yticks(positions, labels)
            detail_axis.invert_yaxis()
            detail_axis.set_xlabel("triple conflicts")
            detail_axis.set_ylabel("primitive slope Δcol/Δrow")
        detail_axis.set_title("Conflict slopes", loc="left", fontsize=12, fontweight="bold")
    detail_axis.spines["top"].set_visible(False)
    detail_axis.spines["right"].set_visible(False)

    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return metrics
