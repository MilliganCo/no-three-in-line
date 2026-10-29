from __future__ import annotations

import concurrent.futures
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from .cp_sat import choose_free_rows, repair_state
from .sat import repair_state_sat
from .state import PairPermutations


@dataclass(frozen=True)
class PortfolioTrial:
    backend: str
    free_row_count: int
    seed: int
    status: str
    free_rows: tuple[int, ...]
    variables: int
    constraints: int
    model_seconds: float
    solve_seconds: float
    state: PairPermutations | None


@dataclass(frozen=True)
class PortfolioResult:
    trials: tuple[PortfolioTrial, ...]
    state: PairPermutations | None

    @property
    def status_counts(self) -> dict[str, int]:
        return dict(Counter(trial.status for trial in self.trials))


def run_trial(
    state: PairPermutations,
    backend: str,
    free_row_count: int,
    seed: int,
    time_limit: float,
    solver_workers: int,
    row_strategy: str = "all-conflicts",
) -> PortfolioTrial:
    free_rows = choose_free_rows(state, free_row_count, seed=seed, strategy=row_strategy)
    if backend == "cp-sat":
        result = repair_state(
            state,
            free_rows,
            time_limit=time_limit,
            seed=seed,
            workers=solver_workers,
        )
        return PortfolioTrial(
            backend,
            free_row_count,
            seed,
            result.status,
            result.free_rows,
            result.variables,
            result.line_constraints,
            result.model_seconds,
            result.solve_seconds,
            result.state,
        )
    if backend == "sat":
        result = repair_state_sat(state, free_rows, time_limit=time_limit)
        return PortfolioTrial(
            backend,
            free_row_count,
            seed,
            result.status,
            result.free_rows,
            result.variables,
            result.clauses,
            result.model_seconds,
            result.solve_seconds,
            result.state,
        )
    raise ValueError(f"unknown portfolio backend {backend}")


def run_lns_portfolio(
    state: PairPermutations,
    *,
    backends: Iterable[str],
    free_row_counts: Iterable[int],
    seeds: Iterable[int],
    time_limit: float,
    solver_workers: int,
    parallel: int,
    row_strategy: str = "all-conflicts",
    callback: Callable[[PortfolioTrial], None] | None = None,
) -> PortfolioResult:
    specs = [
        (state, backend, free_count, seed, time_limit, solver_workers, row_strategy)
        for free_count in free_row_counts
        for backend in backends
        for seed in seeds
    ]
    trials: list[PortfolioTrial] = []
    found = None
    if parallel <= 1:
        iterator = (run_trial(*spec) for spec in specs)
        for trial in iterator:
            trials.append(trial)
            if callback:
                callback(trial)
            if trial.state is not None:
                found = trial.state
                break
        return PortfolioResult(tuple(trials), found)

    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
        futures = [executor.submit(run_trial, *spec) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            trial = future.result()
            trials.append(trial)
            if callback:
                callback(trial)
            if trial.state is not None and found is None:
                found = trial.state
                for pending in futures:
                    pending.cancel()
    trials.sort(key=lambda trial: (trial.free_row_count, trial.backend, trial.seed))
    return PortfolioResult(tuple(trials), found)
