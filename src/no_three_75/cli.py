from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from random import Random

import numpy as np

from .conflicts import analyze_points
from .baseline_cp_sat import solve_baseline
from .analysis import structure_record
from .breakout import breakout_search
from .cp_sat import choose_free_rows, repair_state
from .edge_lns import choose_free_points, repair_state_edges
from .gpu import benchmark_gpu
from .gpu_delta import benchmark_delta, score_two_switches_gpu
from .gpu_device_search import gpu_device_beam_search
from .gpu_exhaustive_beam import gpu_exhaustive_beam_search
from .gpu_search import GpuSearchSnapshot, gpu_population_search
from .gpu_neighborhood import bad_anchor_indices, best_double_switch, best_four_cycle
from .geometry import d4_variants
from .hardware import detect_hardware
from .io import encode_compact, load_candidate, write_candidate
from .search import SearchSnapshot, local_search
from .sat import repair_state_sat
from .sat_assumptions import AssumptionTrial, run_assumption_portfolio
from .portfolio import PortfolioTrial, run_lns_portfolio
from .permutation_cp_sat import solve_permutation_model
from .rct4_sat import solve_rct4_sat
from .lazy_repair import LazyRepairRound, lazy_repair_state
from .sat_distance import solve_distance_ball
from .precompute import benchmark_precomputation
from .multistep_seeds import sample_multistep_expansions
from .seeds import generate_expansion_seeds, generate_projection_seeds
from .state import PairPermutations, points_to_permutations, random_state
from .symmetry_cp_sat import LazyRound, full_rot2_search, full_symmetry_search, lazy_rot2_search, lazy_symmetry_search
from .verifier_a import verify as verify_a
from .verifier_b import verify as verify_b
from .visualize import render_candidate


ROOT = Path(__file__).resolve().parents[2]


def command_verify(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    valid_a, detail_a = verify_a(points, n)
    valid_b, detail_b = verify_b(points, n)
    print(f"verifier_a={'VALID' if valid_a else 'INVALID'}: {detail_a}")
    print(f"verifier_b={'VALID' if valid_b else 'INVALID'}: {detail_b}")
    return 0 if valid_a and valid_b else 1


def command_convert(args: argparse.Namespace) -> int:
    n, points, metadata = load_candidate(args.source, n=args.n)
    state = points_to_permutations(points, n)
    if args.format == "json":
        write_candidate(args.output, n, points, P=list(state.P), Q=list(state.Q), source=metadata)
    else:
        Path(args.output).write_text(encode_compact(points, n, args.symmetry) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(points)} points")
    return 0


def command_transform(args: argparse.Namespace) -> int:
    n, points, metadata = load_candidate(args.source, n=args.n)
    variants = d4_variants(points, n)
    if not 0 <= args.variant < len(variants):
        raise ValueError(f"variant must be between 0 and {len(variants) - 1}")
    transformed = variants[args.variant]
    state = points_to_permutations(transformed, n)
    write_candidate(
        args.output,
        n,
        transformed,
        P=list(state.P),
        Q=list(state.Q),
        transform={"source": str(args.source), "d4_variant": args.variant, "source_metadata": metadata},
    )
    print(f"wrote D4 variant {args.variant} to {args.output}")
    return 0


def command_import_known(args: argparse.Namespace) -> int:
    metadata_path = ROOT / "data" / "known-solutions" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    base = metadata_path.parent
    failures = 0
    for n_text, record in metadata["records"].items():
        n = int(n_text)
        raw_path = base / record["raw"]
        decoded_n, points, raw_metadata = load_candidate(raw_path, n=n)
        ok_a, detail_a = verify_a(points, decoded_n)
        ok_b, detail_b = verify_b(points, decoded_n)
        if not (ok_a and ok_b):
            failures += 1
            print(f"n={n}: INVALID A={detail_a} B={detail_b}")
            continue
        state = points_to_permutations(points, n)
        output = base / f"n{n}.json"
        write_candidate(
            output,
            n,
            points,
            P=list(state.P),
            Q=list(state.Q),
            symmetry=record["symmetry"],
            author=record["author"],
            discovery_date=record["date"],
            source=metadata["source"] + Path(record["raw"]).name.replace(".txt", ".few"),
            compact=raw_metadata["compact"],
            verification={"determinant": detail_a, "normalized_slopes": detail_b},
        )
        print(f"n={n}: VALID -> {output.name}")
    return 1 if failures else 0


def command_analyze(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    metrics = analyze_points(points)
    print(json.dumps({
        "n": n,
        "points": len(points),
        "triples": metrics.triples,
        "violating_lines": metrics.violating_lines,
        "max_line_occupancy": metrics.max_line_occupancy,
        "max_point_badness": metrics.max_point_badness,
        "slope_histogram": {str(key): value for key, value in metrics.slope_histogram.items()},
    }, indent=2))
    return 0


def command_structure(args: argparse.Namespace) -> int:
    records = []
    for candidate in args.candidates:
        n, points, _ = load_candidate(candidate)
        record = structure_record(n, points, include_neighborhood=args.neighborhood)
        record["candidate"] = str(candidate)
        records.append(record)
    payload = {"records": records}
    output = json.dumps(payload, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output, end="")
    return 0


def command_minima_db(args: argparse.Namespace) -> int:
    groups: dict[str, dict[str, object]] = {}
    scanned = 0
    for path in Path(args.root).rglob("*.json"):
        try:
            n, points, _ = load_candidate(path)
            if n != args.n:
                continue
            record = structure_record(n, points)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        scanned += 1
        if int(record["triples"]) > args.max_score:
            continue
        key = str(record["canonical_d4_sha256"])
        if key not in groups:
            groups[key] = {
                "canonical_d4_sha256": key,
                "score": record["triples"],
                "cycle_lengths": record["bipartite_cycle_lengths"],
                "d4_stabilizer_size": record["d4_stabilizer_size"],
                "representative": str(path),
                "occurrences": [],
            }
        groups[key]["occurrences"].append(str(path))
    records = sorted(groups.values(), key=lambda item: (item["score"], item["canonical_d4_sha256"]))
    payload = {
        "n": args.n,
        "max_score": args.max_score,
        "scanned_candidates": scanned,
        "d4_isomorphism_classes": len(records),
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


def command_visualize(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    metrics = render_candidate(n, points, args.output, title=args.title, history=args.history)
    print(f"wrote {args.output}: n={n} points={len(points)} E={metrics.triples}")
    return 0


def command_gpu_benchmark(args: argparse.Namespace) -> int:
    states = [random_state(args.n, Random(args.seed + index)) for index in range(args.population)]
    p = np.asarray([state.P for state in states], dtype=np.int32)
    q = np.asarray([state.Q for state in states], dtype=np.int32)
    result = benchmark_gpu(p, q, repeats=args.repeats)
    payload = {
        "device": result.device,
        "n": result.n,
        "population": result.population,
        "cpu_seconds": result.cpu_seconds,
        "gpu_end_to_end_seconds": result.gpu_end_to_end_seconds,
        "gpu_kernel_seconds": result.gpu_kernel_seconds,
        "end_to_end_speedup": result.end_to_end_speedup,
        "kernel_speedup": result.kernel_speedup,
        "exact_match": result.exact_match,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.exact_match else 1


def command_hardware(args: argparse.Namespace) -> int:
    print(json.dumps(detect_hardware().as_dict(), indent=2))
    return 0


def command_precompute_benchmark(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    result = benchmark_precomputation(n, points)
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"wrote {args.output}")
    print(output, end="")
    return 0


def command_gpu_delta_benchmark(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    state = points_to_permutations(points, n)
    p = np.asarray(state.P, dtype=np.int32)
    q = np.asarray(state.Q, dtype=np.int32)
    from .accelerated import score_and_badness

    score, badness = score_and_badness(p, q)
    anchors = bad_anchor_indices(state, args.anchors)
    results = []
    for anchor in anchors:
        result = benchmark_delta(p, q, anchor, score, badness, repeats=args.repeats)
        results.append({
            "anchor": anchor,
            "cpu_seconds": result.cpu_seconds,
            "gpu_seconds": result.gpu_seconds,
            "speedup": result.speedup,
            "exact_match": result.exact_match,
        })
    print(json.dumps({"n": n, "score": int(score), "repeats": args.repeats, "anchors": results}, indent=2))
    return 0 if all(result["exact_match"] for result in results) else 1


def command_gpu_search(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.initial, n=args.n)
    initial = points_to_permutations(points, n)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"gpu-seed-{args.seed}.jsonl"
    last_reported = {"score": None}
    initial_elites = None
    if args.resume_population:
        with np.load(args.resume_population) as checkpoint_data:
            initial_elites = np.stack((checkpoint_data["P"], checkpoint_data["Q"]), axis=2)
    candidate_paths = list(args.additional_initial)
    for directory in args.additional_initial_dir:
        candidate_paths.extend(str(path) for path in sorted(Path(directory).glob("seed-*.json")))
    additional = []
    for candidate in candidate_paths:
        extra_n, extra_points, _ = load_candidate(candidate, n=n)
        extra_state = points_to_permutations(extra_points, extra_n)
        additional.append(np.column_stack((extra_state.P, extra_state.Q)).astype(np.int32))
    if additional:
        extra_array = np.asarray(additional, dtype=np.int32)
        initial_elites = extra_array if initial_elites is None else np.concatenate((initial_elites, extra_array), axis=0)

    def checkpoint(snapshot: GpuSearchSnapshot) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": snapshot.generation,
            "elapsed_seconds": snapshot.elapsed_seconds,
            "evaluations": snapshot.evaluations,
            "generation_best": snapshot.generation_best,
            "best_score": snapshot.best_score,
            "elite_scores": list(snapshot.elite_scores),
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        if args.checkpoint_every and snapshot.generation % args.checkpoint_every == 0:
            np.savez_compressed(
                output_dir / f"population-gen-{snapshot.generation:06d}.npz",
                P=snapshot.elite_values[:, :, 0],
                Q=snapshot.elite_values[:, :, 1],
                scores=np.asarray(snapshot.elite_scores, dtype=np.int32),
                generation=np.asarray(snapshot.generation, dtype=np.int64),
                seed=np.asarray(args.seed, dtype=np.int64),
            )
        if snapshot.best_score != last_reported["score"]:
            last_reported["score"] = snapshot.best_score
            write_candidate(
                output_dir / f"best-gpu-seed-{args.seed}.json",
                n,
                snapshot.state.points(),
                P=list(snapshot.state.P),
                Q=list(snapshot.state.Q),
                conflicts=snapshot.best_score,
                generation=snapshot.generation,
                evaluations=snapshot.evaluations,
            )
            print(
                f"generation={snapshot.generation} best={snapshot.best_score} "
                f"evaluations={snapshot.evaluations} elapsed={snapshot.elapsed_seconds:.3f}s",
                flush=True,
            )

    if args.engine == "beam":
        result = gpu_exhaustive_beam_search(
            initial,
            seed=args.seed,
            generations=args.generations,
            elite_size=args.elite,
            energy_window=args.energy_window,
            initial_elites=initial_elites,
            global_every=args.global_every,
            report_every=args.report_every,
            callback=checkpoint,
        )
    elif args.engine == "device":
        result = gpu_device_beam_search(
            initial,
            seed=args.seed,
            generations=args.generations,
            population=args.population,
            elite_size=args.elite,
            max_strength=args.max_strength,
            energy_window=args.energy_window,
            initial_elites=initial_elites,
            report_every=args.report_every,
            bad_bias=args.bad_bias,
            callback=checkpoint,
        )
    else:
        result = gpu_population_search(
            initial,
            seed=args.seed,
            generations=args.generations,
            population=args.population,
            elite_size=args.elite,
            max_strength=args.max_strength,
            energy_window=args.energy_window,
            crossover_probability=args.crossover_probability,
            initial_elites=initial_elites,
            callback=checkpoint,
        )
    write_candidate(
        output_dir / f"result-gpu-seed-{args.seed}-E{result.score}.json",
        n,
        result.state.points(),
        P=list(result.state.P),
        Q=list(result.state.Q),
        conflicts=result.score,
        generations=result.generations,
        evaluations=result.evaluations,
        elapsed_seconds=result.elapsed_seconds,
    )
    print(json.dumps({
        "best_score": result.score,
        "generations": result.generations,
        "evaluations": result.evaluations,
        "elapsed_seconds": result.elapsed_seconds,
        "states_per_second": result.evaluations / max(result.elapsed_seconds, 1e-9),
    }, indent=2))
    return 0 if result.score == 0 else 1


def command_gpu_four_cycle(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    best_state = points_to_permutations(points, n)
    best_score = analyze_points(best_state.points()).triples
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_evaluated = 0
    total_legal = 0
    started = time.perf_counter()
    for round_index in range(1, args.rounds + 1):
        improved = False
        anchors = bad_anchor_indices(best_state, args.anchor_offset + args.anchors)[args.anchor_offset:]
        for anchor in anchors:
            result = best_four_cycle(best_state, anchor, batch_size=args.batch_size)
            total_evaluated += result.evaluated
            total_legal += result.legal
            payload = {
                "round": round_index,
                "anchor": anchor,
                "score": result.score,
                "best_before": best_score,
                "evaluated": result.evaluated,
                "legal": result.legal,
                "seconds": result.elapsed_seconds,
            }
            print(json.dumps(payload), flush=True)
            if result.score < best_score:
                best_state = result.state
                best_score = result.score
                improved = True
                write_candidate(
                    output_dir / f"best-four-cycle-E{best_score}.json",
                    n,
                    best_state.points(),
                    P=list(best_state.P),
                    Q=list(best_state.Q),
                    gpu_four_cycle=payload,
                )
                break
        if not improved or best_score == 0:
            break
    summary = {
        "best_score": best_score,
        "rounds": round_index,
        "evaluated": total_evaluated,
        "legal": total_legal,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_candidate(
        output_dir / f"result-four-cycle-E{best_score}.json",
        n,
        best_state.points(),
        P=list(best_state.P),
        Q=list(best_state.Q),
        gpu_four_cycle_search=summary,
    )
    print(json.dumps(summary, indent=2))
    return 0 if best_score == 0 else 1


def command_gpu_double_switch(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    best_state = points_to_permutations(points, n)
    best_score = analyze_points(best_state.points()).triples
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_evaluated = 0
    total_legal = 0
    started = time.perf_counter()
    anchors = bad_anchor_indices(best_state, args.anchor_offset + args.anchors)[args.anchor_offset:]
    for anchor in anchors:
        result = best_double_switch(best_state, anchor, batch_size=args.batch_size)
        total_evaluated += result.evaluated
        total_legal += result.legal
        payload = {
            "anchor": anchor,
            "score": result.score,
            "best_before": best_score,
            "evaluated": result.evaluated,
            "legal": result.legal,
            "seconds": result.elapsed_seconds,
        }
        print(json.dumps(payload), flush=True)
        if result.score < best_score:
            best_state = result.state
            best_score = result.score
            write_candidate(
                output_dir / f"best-double-switch-E{best_score}.json",
                n,
                best_state.points(),
                P=list(best_state.P),
                Q=list(best_state.Q),
                gpu_double_switch=payload,
            )
            break
    summary = {
        "best_score": best_score,
        "evaluated": total_evaluated,
        "legal": total_legal,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_candidate(
        output_dir / f"result-double-switch-E{best_score}.json",
        n,
        best_state.points(),
        P=list(best_state.P),
        Q=list(best_state.Q),
        gpu_double_switch_search=summary,
    )
    print(json.dumps(summary, indent=2))
    return 0 if best_score == 0 else 1


def command_cp_sat(args: argparse.Namespace) -> int:
    hint_points = None
    if args.hint:
        hint_n, hint_points, _ = load_candidate(args.hint, n=args.n)
        if hint_n != args.n:
            raise ValueError("hint size does not match the CP-SAT instance")
    result = solve_baseline(
        args.n,
        time_limit=args.time_limit,
        seed=args.seed,
        workers=args.workers,
        log_search=args.log_search,
        hint_points=hint_points,
    )
    payload = {
        "n": args.n,
        "status": result.status,
        "variables": result.variables,
        "line_constraints": result.line_constraints,
        "model_seconds": result.model_seconds,
        "solve_seconds": result.solve_seconds,
        "conflicts": result.conflicts,
        "branches": result.branches,
    }
    print(json.dumps(payload, indent=2))
    if result.points is not None:
        state = points_to_permutations(list(result.points), args.n)
        write_candidate(
            args.output,
            args.n,
            list(result.points),
            P=list(state.P),
            Q=list(state.Q),
            solver="CP-SAT direct baseline",
            run=payload,
        )
        print(f"VALID witness written to {args.output}")
        return 0
    return 1


def command_perm_cp_sat(args: argparse.Namespace) -> int:
    hint = None
    if args.hint:
        hint_n, hint_points, _ = load_candidate(args.hint, n=args.n)
        if hint_n != args.n:
            raise ValueError("hint size does not match the permutation instance")
        hint = points_to_permutations(hint_points, hint_n)
    result = solve_permutation_model(
        args.n,
        time_limit=args.time_limit,
        seed=args.seed,
        workers=args.workers,
        hint=hint,
        log_search=args.log_search,
    )
    payload = {
        "n": args.n,
        "status": result.status,
        "variables": result.variables,
        "collinearity_constraints": result.collinearity_constraints,
        "model_seconds": result.model_seconds,
        "solve_seconds": result.solve_seconds,
        "conflicts": result.conflicts,
        "branches": result.branches,
    }
    print(json.dumps(payload, indent=2))
    if result.state is not None:
        write_candidate(
            args.output,
            args.n,
            result.state.points(),
            P=list(result.state.P),
            Q=list(result.state.Q),
            solver="CP-SAT permutation model",
            run=payload,
        )
        print(f"VALID witness written to {args.output}")
        return 0
    return 1


def command_rct4_sat(args: argparse.Namespace) -> int:
    hint_points = None
    if args.hint:
        hint_n, hint_points, _ = load_candidate(args.hint, n=args.n)
        if hint_n != args.n:
            raise ValueError("hint size does not match the rct4 SAT instance")
    result = solve_rct4_sat(
        args.n,
        time_limit=args.time_limit,
        solver_name=args.solver,
        encoding=args.encoding,
        hint_points=hint_points,
        force_hint=args.force_hint,
    )
    payload = {
        "n": args.n,
        "status": result.status,
        "orbit_variables": result.orbit_variables,
        "variables": result.variables,
        "clauses": result.clauses,
        "line_constraints": result.line_constraints,
        "model_seconds": result.model_seconds,
        "solve_seconds": result.solve_seconds,
    }
    print(json.dumps(payload, indent=2))
    if result.points is not None:
        state = points_to_permutations(list(result.points), args.n)
        write_candidate(
            args.output,
            args.n,
            list(result.points),
            P=list(state.P),
            Q=list(state.Q),
            solver="rct4 SAT",
            run=payload,
        )
        print(f"VALID witness written to {args.output}")
        return 0
    return 1


def command_seeds(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.source)
    if n != args.target + 1:
        raise ValueError(f"projection currently requires source n=target+1, got {n} -> {args.target}")
    source = points_to_permutations(points, n)
    experiment = generate_projection_seeds(
        source,
        keep=args.keep,
        include_d4=not args.no_d4,
        include_decompositions=not args.single_decomposition,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(experiment.best):
        write_candidate(
            output_dir / f"seed-{index:04d}-E{record.score}.json",
            args.target,
            record.state.points(),
            P=list(record.state.P),
            Q=list(record.state.Q),
            initial_source=str(args.source),
            projection={
                "removed_row": record.removed_row,
                "removed_column": record.removed_column,
                "d4_variant": record.variant,
                "matching_decomposition": record.decomposition,
            },
            conflicts=record.score,
        )
    summary = {
        "source": str(args.source),
        "target": args.target,
        "evaluated": experiment.evaluated,
        "unique": experiment.unique,
        "histogram": {str(key): value for key, value in experiment.histogram.items()},
        "best_scores": [record.score for record in experiment.best],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def command_expand_seeds(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.source)
    if n != args.target - 1:
        raise ValueError(f"expansion requires source n=target-1, got {n} -> {args.target}")
    source = points_to_permutations(points, n)
    experiment = generate_expansion_seeds(
        source,
        keep=args.keep,
        include_d4=not args.no_d4,
        include_decompositions=not args.single_decomposition,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(experiment.best):
        write_candidate(
            output_dir / f"seed-{index:04d}-E{record.score}.json",
            args.target,
            record.state.points(),
            P=list(record.state.P),
            Q=list(record.state.Q),
            initial_source=str(args.source),
            expansion={
                "break_p": record.break_p,
                "break_q": record.break_q,
                "d4_variant": record.variant,
                "matching_decomposition": record.decomposition,
            },
            conflicts=record.score,
        )
    summary = {
        "source": str(args.source),
        "target": args.target,
        "evaluated": experiment.evaluated,
        "unique": experiment.unique,
        "histogram": {str(key): value for key, value in experiment.histogram.items()},
        "best_scores": [record.score for record in experiment.best],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def command_sample_expand(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.source)
    source = points_to_permutations(points, n)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    def report(evaluated: int, best_score: int) -> None:
        print(f"evaluated={evaluated} best={best_score}", flush=True)

    experiment = sample_multistep_expansions(
        source,
        args.target,
        samples=args.samples,
        keep=args.keep,
        seed=args.seed,
        batch_size=args.batch_size,
        backend=args.backend,
        allow_interior_insertions=args.interior,
        callback=report,
    )
    for index, record in enumerate(experiment.best):
        steps = [
            {
                "inserted_row": step.inserted_row,
                "inserted_column": step.inserted_column,
                "break_p": step.break_p,
                "break_q": step.break_q,
            }
            for step in record.steps
        ]
        write_candidate(
            output_dir / f"seed-{index:04d}-E{record.score}.json",
            args.target,
            record.state.points(),
            P=list(record.state.P),
            Q=list(record.state.Q),
            initial_source=str(args.source),
            source_variant=record.source_variant,
            matching_decomposition=record.decomposition,
            expansion_steps=steps,
            conflicts=record.score,
        )
    summary = {
        "source": str(args.source),
        "target": args.target,
        "samples": experiment.evaluated,
        "seed": args.seed,
        "backend": args.backend,
        "interior_insertions": args.interior,
        "histogram": {str(key): value for key, value in experiment.histogram.items()},
        "best_scores": [record.score for record in experiment.best],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def command_search(args: argparse.Namespace) -> int:
    if args.initial:
        initial_n, points, _ = load_candidate(args.initial, n=args.n)
        initial = points_to_permutations(points, initial_n)
    else:
        initial = None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"search-seed-{args.seed}.jsonl"

    def checkpoint(snapshot: SearchSnapshot) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration": snapshot.iteration,
            "elapsed_seconds": snapshot.elapsed_seconds,
            "score": snapshot.score,
            "best_score": snapshot.best_score,
            "move": {
                "layer_a": snapshot.layer,
                "row_a": snapshot.row_a,
                "layer_b": snapshot.partner_layer,
                "row_b": snapshot.row_b,
            },
            "P": list(snapshot.state.P),
            "Q": list(snapshot.state.Q),
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        write_candidate(
            output_dir / f"best-seed-{args.seed}.json",
            args.n,
            snapshot.state.points(),
            P=list(snapshot.state.P),
            Q=list(snapshot.state.Q),
            conflicts=snapshot.best_score,
            search_seed=args.seed,
            iteration=snapshot.iteration,
        )
        print(f"iteration={snapshot.iteration} best={snapshot.best_score} elapsed={snapshot.elapsed_seconds:.3f}s", flush=True)

    evaluator = score_two_switches_gpu if args.backend == "cuda-delta" else None
    search_kwargs = {}
    if evaluator is not None:
        search_kwargs["move_evaluator"] = evaluator
    result = local_search(
        args.n,
        seed=args.seed,
        iterations=args.iterations,
        initial=initial,
        tabu_tenure=args.tabu,
        plateau=args.plateau,
        checkpoint=checkpoint,
        **search_kwargs,
    )
    write_candidate(
        output_dir / f"result-seed-{args.seed}-E{result.score}.json",
        args.n,
        result.state.points(),
        P=list(result.state.P),
        Q=list(result.state.Q),
        conflicts=result.score,
        search_seed=result.seed,
        iterations=result.iterations,
        elapsed_seconds=result.elapsed_seconds,
        moves_evaluated=result.moves_evaluated,
        host=platform.node(),
    )
    print(json.dumps({
        "best_score": result.score,
        "iterations": result.iterations,
        "elapsed_seconds": result.elapsed_seconds,
        "moves_evaluated": result.moves_evaluated,
        "moves_per_second": result.moves_evaluated / max(result.elapsed_seconds, 1e-9),
    }, indent=2))
    return 0


def command_breakout(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.initial, n=args.n)
    initial = points_to_permutations(points, n)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"breakout-seed-{args.seed}.jsonl"

    def checkpoint(snapshot: SearchSnapshot) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration": snapshot.iteration,
            "elapsed_seconds": snapshot.elapsed_seconds,
            "best_score": snapshot.best_score,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        write_candidate(
            output_dir / f"best-breakout-seed-{args.seed}.json",
            n,
            snapshot.state.points(),
            P=list(snapshot.state.P),
            Q=list(snapshot.state.Q),
            conflicts=snapshot.best_score,
            iteration=snapshot.iteration,
        )
        print(f"iteration={snapshot.iteration} best={snapshot.best_score}", flush=True)

    evaluator = score_two_switches_gpu if args.backend == "cuda-delta" else None
    search_kwargs = {}
    if evaluator is not None:
        search_kwargs["move_evaluator"] = evaluator
    result = breakout_search(
        initial,
        seed=args.seed,
        iterations=args.iterations,
        plateau=args.plateau,
        weight_increment=args.weight_increment,
        max_weighted_lines=args.max_weighted_lines,
        checkpoint=checkpoint,
        **search_kwargs,
    )
    summary = {
        "best_score": result.score,
        "iterations": result.iterations,
        "elapsed_seconds": result.elapsed_seconds,
        "moves_evaluated": result.moves_evaluated,
        "moves_per_second": result.moves_evaluated / max(result.elapsed_seconds, 1e-9),
        "breakouts": result.breakouts,
        "weighted_lines": result.weighted_lines,
    }
    write_candidate(
        output_dir / f"result-breakout-seed-{args.seed}-E{result.score}.json",
        n,
        result.state.points(),
        P=list(result.state.P),
        Q=list(result.state.Q),
        conflicts=result.score,
        breakout=summary,
    )
    print(json.dumps(summary, indent=2))
    return 0 if result.score == 0 else 1


def command_repair(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    state = points_to_permutations(points, n)
    free_rows = choose_free_rows(state, args.free_rows, seed=args.seed)
    result = repair_state(
        state,
        free_rows,
        time_limit=args.time_limit,
        seed=args.seed,
        workers=args.workers,
        log_search=args.log_search,
    )
    summary = {
        "status": result.status,
        "free_rows": list(result.free_rows),
        "variables": result.variables,
        "line_constraints": result.line_constraints,
        "model_seconds": result.model_seconds,
        "solve_seconds": result.solve_seconds,
        "conflicts": result.conflicts,
        "branches": result.branches,
    }
    print(json.dumps(summary, indent=2))
    if result.state is not None:
        write_candidate(
            args.output,
            n,
            result.state.points(),
            P=list(result.state.P),
            Q=list(result.state.Q),
            repair=summary,
            source_candidate=str(args.candidate),
        )
        print(f"VALID repair written to {args.output}")
        return 0
    return 1


def command_lazy_repair(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    state = points_to_permutations(points, n)
    free_rows = choose_free_rows(
        state,
        args.free_rows,
        seed=args.seed,
        strategy=args.row_strategy,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = output_dir / f"lazy-repair-seed-{args.seed}.jsonl"
    best_key: dict[str, tuple[int, int, int, int] | None] = {"value": None}

    def report(event: LazyRepairRound) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": event.round_index,
            "status": event.status,
            "triples": event.metrics.triples if event.metrics else None,
            "violating_lines": event.metrics.violating_lines if event.metrics else None,
            "new_constraints": event.new_constraints,
            "total_constraints": event.total_constraints,
            "hamming_distance": event.hamming_distance,
            "solve_seconds": event.solve_seconds,
        }
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        print(json.dumps(payload), flush=True)
        if event.state is not None and event.metrics is not None:
            key = event.metrics.lexicographic
            if best_key["value"] is None or key < best_key["value"]:
                best_key["value"] = key
                write_candidate(
                    output_dir / f"best-seed-{args.seed}-E{event.metrics.triples}.json",
                    n,
                    event.state.points(),
                    P=list(event.state.P),
                    Q=list(event.state.Q),
                    lazy_repair=payload,
                )

    result = lazy_repair_state(
        state,
        free_rows,
        max_rounds=args.rounds,
        time_limit_per_round=args.time_limit,
        seed=args.seed,
        workers=args.workers,
        minimize_hamming=not args.no_hamming,
        callback=report,
    )
    summary = {
        "status": result.status,
        "rounds": result.rounds,
        "constraints": result.constraints,
        "elapsed_seconds": result.elapsed_seconds,
        "best_triples": result.metrics.triples,
        "free_rows": free_rows,
    }
    write_candidate(
        output_dir / f"result-seed-{args.seed}-E{result.metrics.triples}.json",
        n,
        result.state.points(),
        P=list(result.state.P),
        Q=list(result.state.Q),
        lazy_repair=summary,
    )
    print(json.dumps(summary, indent=2))
    return 0 if result.status == "VALID" else 1


def command_sat_repair(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    state = points_to_permutations(points, n)
    free_rows = choose_free_rows(state, args.free_rows, seed=args.seed)
    result = repair_state_sat(
        state,
        free_rows,
        time_limit=args.time_limit,
        solver_name=args.solver,
        encoding=args.encoding,
    )
    payload = {
        "status": result.status,
        "free_rows": list(result.free_rows),
        "variables": result.variables,
        "clauses": result.clauses,
        "line_constraints": result.line_constraints,
        "model_seconds": result.model_seconds,
        "solve_seconds": result.solve_seconds,
        "solver": args.solver,
        "encoding": args.encoding,
    }
    print(json.dumps(payload, indent=2))
    if result.state is not None:
        write_candidate(
            args.output,
            n,
            result.state.points(),
            P=list(result.state.P),
            Q=list(result.state.Q),
            sat_repair=payload,
            source_candidate=str(args.candidate),
        )
        print(f"VALID SAT repair written to {args.output}")
        return 0
    return 1


def command_sat_distance(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    state = points_to_permutations(points, n)
    result = solve_distance_ball(
        state,
        max_removed_points=args.max_removed,
        time_limit=args.time_limit,
        solver_name=args.solver,
        encoding=args.encoding,
    )
    payload = {
        "n": n,
        "status": result.status,
        "max_removed_points": result.max_removed_points,
        "symmetric_difference_radius": 2 * result.max_removed_points,
        "variables": result.variables,
        "clauses": result.clauses,
        "line_constraints": result.line_constraints,
        "model_seconds": result.model_seconds,
        "solve_seconds": result.solve_seconds,
    }
    print(json.dumps(payload, indent=2))
    if result.state is not None:
        write_candidate(
            args.output,
            n,
            result.state.points(),
            P=list(result.state.P),
            Q=list(result.state.Q),
            distance_sat=payload,
        )
        print(f"VALID distance-ball witness written to {args.output}")
        return 0
    return 1


def command_edge_repair(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    state = points_to_permutations(points, n)
    free_points = choose_free_points(state, args.free_points, seed=args.seed)
    result = repair_state_edges(
        state,
        free_points,
        time_limit=args.time_limit,
        seed=args.seed,
        workers=args.workers,
        log_search=args.log_search,
    )
    payload = {
        "status": result.status,
        "free_point_count": len(result.free_points),
        "free_points": [list(point) for point in result.free_points],
        "active_rows": result.active_rows,
        "active_columns": result.active_columns,
        "variables": result.variables,
        "line_constraints": result.line_constraints,
        "model_seconds": result.model_seconds,
        "solve_seconds": result.solve_seconds,
        "conflicts": result.conflicts,
        "branches": result.branches,
    }
    print(json.dumps(payload, indent=2))
    if result.state is not None:
        write_candidate(
            args.output,
            n,
            result.state.points(),
            P=list(result.state.P),
            Q=list(result.state.Q),
            edge_repair=payload,
            source_candidate=str(args.candidate),
        )
        print(f"VALID edge repair written to {args.output}")
        return 0
    return 1


def command_lns_portfolio(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    state = points_to_permutations(points, n)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = output_dir / "portfolio.jsonl"

    def report(trial: PortfolioTrial) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": trial.backend,
            "free_row_count": trial.free_row_count,
            "seed": trial.seed,
            "status": trial.status,
            "free_rows": list(trial.free_rows),
            "variables": trial.variables,
            "constraints": trial.constraints,
            "model_seconds": trial.model_seconds,
            "solve_seconds": trial.solve_seconds,
        }
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        print(json.dumps(payload), flush=True)

    result = run_lns_portfolio(
        state,
        backends=args.backends,
        free_row_counts=args.free_rows,
        seeds=args.seeds,
        time_limit=args.time_limit,
        solver_workers=args.solver_workers,
        parallel=args.parallel,
        row_strategy=args.row_strategy,
        callback=report,
    )
    summary = {
        "trials": len(result.trials),
        "status_counts": result.status_counts,
        "witness": result.state is not None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if result.state is not None:
        write_candidate(
            output_dir / "n75-portfolio-witness.json",
            n,
            result.state.points(),
            P=list(result.state.P),
            Q=list(result.state.Q),
            portfolio=summary,
        )
        return 0
    return 1


def command_sat_assumptions(args: argparse.Namespace) -> int:
    n, points, _ = load_candidate(args.candidate, n=args.n)
    state = points_to_permutations(points, n)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = output_dir / "portfolio.jsonl"

    def report(trial: AssumptionTrial) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "free_row_count": trial.free_row_count,
            "seed": trial.seed,
            "free_rows": list(trial.free_rows),
            "status": trial.status,
            "assumptions": trial.assumptions,
            "solve_seconds": trial.solve_seconds,
        }
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        print(json.dumps(payload), flush=True)

    result = run_assumption_portfolio(
        state,
        free_row_counts=args.free_rows,
        seeds=args.seeds,
        time_limit=args.time_limit,
        solver_name=args.solver,
        encoding=args.encoding,
        row_strategy=args.row_strategy,
        callback=report,
    )
    summary = {
        "n": n,
        "solver": args.solver,
        "encoding": args.encoding,
        "variables": result.variables,
        "clauses": result.clauses,
        "line_constraints": result.line_constraints,
        "model_seconds": result.model_seconds,
        "solver_setup_seconds": result.solver_setup_seconds,
        "trials": len(result.trials),
        "status_counts": {
            status: sum(trial.status == status for trial in result.trials)
            for status in sorted({trial.status for trial in result.trials})
        },
        "witness": result.state is not None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if result.state is not None:
        write_candidate(
            output_dir / "n75-assumption-witness.json",
            n,
            result.state.points(),
            P=list(result.state.P),
            Q=list(result.state.Q),
            assumption_portfolio=summary,
        )
        return 0
    return 1


def command_sym_search(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"rct4-n{args.n}-seed{args.seed}.jsonl"
    best_reported: dict[str, tuple[int, int, int, int] | None] = {"key": None}

    def report(event: LazyRound) -> None:
        payload = {
            "round": event.round_index,
            "status": event.status,
            "triples": event.metrics.triples if event.metrics else None,
            "violating_lines": event.metrics.violating_lines if event.metrics else None,
            "new_constraints": event.new_constraints,
            "total_constraints": event.total_constraints,
            "solve_seconds": event.solve_seconds,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        print(json.dumps(payload), flush=True)
        if event.points is not None:
            write_candidate(
                output_dir / f"latest-n{args.n}-seed{args.seed}.json",
                args.n,
                event.points,
                symmetry="rct4" if args.n % 2 else "rot4",
                lazy_round=payload,
            )
            key = event.metrics.lexicographic if event.metrics else None
            if key is not None and (best_reported["key"] is None or key < best_reported["key"]):
                best_reported["key"] = key
                write_candidate(
                    output_dir / f"best-n{args.n}-seed{args.seed}-E{event.metrics.triples}.json",
                    args.n,
                    event.points,
                    symmetry="rct4" if args.n % 2 else "rot4",
                    lazy_round=payload,
                )

    hint_points = None
    if args.hint:
        hint_n, hint_points, _ = load_candidate(args.hint, n=args.n)
        if hint_n != args.n:
            raise ValueError("hint size does not match the lazy symmetry instance")
    result = lazy_symmetry_search(
        args.n,
        max_rounds=args.rounds,
        time_limit_per_round=args.time_limit,
        seed=args.seed,
        workers=args.workers,
        hint_points=hint_points,
        minimize_hint_distance=args.minimize_hint_distance,
        callback=report,
    )
    summary = {
        "status": result.status,
        "rounds": result.rounds,
        "constraints": result.constraints,
        "elapsed_seconds": result.elapsed_seconds,
        "best_triples": result.metrics.triples if result.metrics else None,
    }
    print(json.dumps(summary, indent=2))
    if result.points is not None:
        write_candidate(
            output_dir / f"best-n{args.n}-seed{args.seed}-E{result.metrics.triples}.json",
            args.n,
            result.points,
            symmetry="rct4" if args.n % 2 else "rot4",
            lazy_search=summary,
        )
    return 0 if result.status == "VALID" else 1


def command_rot2_search(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"rot2-n{args.n}-seed{args.seed}.jsonl"

    def report(event: LazyRound) -> None:
        payload = {
            "round": event.round_index,
            "status": event.status,
            "triples": event.metrics.triples if event.metrics else None,
            "violating_lines": event.metrics.violating_lines if event.metrics else None,
            "new_constraints": event.new_constraints,
            "total_constraints": event.total_constraints,
            "solve_seconds": event.solve_seconds,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        print(json.dumps(payload), flush=True)
        if event.points is not None:
            write_candidate(
                output_dir / f"latest-n{args.n}-seed{args.seed}.json",
                args.n,
                event.points,
                symmetry="rot2",
                lazy_round=payload,
            )

    result = lazy_rot2_search(
        args.n,
        max_rounds=args.rounds,
        time_limit_per_round=args.time_limit,
        seed=args.seed,
        workers=args.workers,
        callback=report,
    )
    summary = {
        "status": result.status,
        "rounds": result.rounds,
        "constraints": result.constraints,
        "elapsed_seconds": result.elapsed_seconds,
        "best_triples": result.metrics.triples if result.metrics else None,
    }
    print(json.dumps(summary, indent=2))
    if result.points is not None:
        write_candidate(
            output_dir / f"best-n{args.n}-seed{args.seed}-E{result.metrics.triples}.json",
            args.n,
            result.points,
            symmetry="rot2",
            lazy_search=summary,
        )
    return 0 if result.status == "VALID" else 1


def command_sym_full(args: argparse.Namespace) -> int:
    hint_points = None
    if args.hint:
        hint_n, hint_points, _ = load_candidate(args.hint, n=args.n)
        if hint_n != args.n:
            raise ValueError("hint size does not match symmetry instance")
    result = full_symmetry_search(
        args.n,
        time_limit=args.time_limit,
        seed=args.seed,
        workers=args.workers,
        hint_points=hint_points,
        log_search=args.log_search,
    )
    payload = {
        "n": args.n,
        "status": result.status,
        "variables": result.variables,
        "line_constraints": result.line_constraints,
        "model_seconds": result.model_seconds,
        "solve_seconds": result.solve_seconds,
        "conflicts": result.conflicts,
        "branches": result.branches,
    }
    print(json.dumps(payload, indent=2))
    if result.points is not None:
        write_candidate(args.output, args.n, list(result.points), symmetry="rct4" if args.n % 2 else "rot4", run=payload)
        print(f"VALID symmetry witness written to {args.output}")
        return 0
    return 1


def command_rot2_full(args: argparse.Namespace) -> int:
    hint_points = None
    if args.hint:
        hint_n, hint_points, _ = load_candidate(args.hint, n=args.n)
        if hint_n != args.n:
            raise ValueError("hint size does not match rot2 instance")
    result = full_rot2_search(
        args.n,
        time_limit=args.time_limit,
        seed=args.seed,
        workers=args.workers,
        hint_points=hint_points,
        log_search=args.log_search,
    )
    payload = {
        "n": args.n,
        "symmetry": "rot2",
        "status": result.status,
        "variables": result.variables,
        "line_constraints": result.line_constraints,
        "model_seconds": result.model_seconds,
        "solve_seconds": result.solve_seconds,
        "conflicts": result.conflicts,
        "branches": result.branches,
    }
    print(json.dumps(payload, indent=2))
    if result.points is not None:
        write_candidate(args.output, args.n, list(result.points), symmetry="rot2", run=payload)
        print(f"VALID rot2 witness written to {args.output}")
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="no3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("candidate")
    verify.add_argument("--n", type=int)
    verify.set_defaults(function=command_verify)

    convert = subparsers.add_parser("convert")
    convert.add_argument("source")
    convert.add_argument("output")
    convert.add_argument("--n", type=int)
    convert.add_argument("--format", choices=("json", "compact"), default="json")
    convert.add_argument("--symmetry", default="iden")
    convert.set_defaults(function=command_convert)

    transform = subparsers.add_parser("transform")
    transform.add_argument("source")
    transform.add_argument("output")
    transform.add_argument("--n", type=int)
    transform.add_argument("--variant", type=int, required=True)
    transform.set_defaults(function=command_transform)

    import_known = subparsers.add_parser("import-known")
    import_known.set_defaults(function=command_import_known)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("candidate")
    analyze.add_argument("--n", type=int)
    analyze.set_defaults(function=command_analyze)

    structure = subparsers.add_parser("structure")
    structure.add_argument("candidates", nargs="+")
    structure.add_argument("--neighborhood", action="store_true")
    structure.add_argument("--output")
    structure.set_defaults(function=command_structure)

    minima_db = subparsers.add_parser("minima-db")
    minima_db.add_argument("--root", default="data/runs")
    minima_db.add_argument("--n", type=int, default=75)
    minima_db.add_argument("--max-score", type=int, default=20)
    minima_db.add_argument("--output", default="data/best/plateau-database.json")
    minima_db.set_defaults(function=command_minima_db)

    visualize = subparsers.add_parser("visualize")
    visualize.add_argument("candidate")
    visualize.add_argument("--n", type=int)
    visualize.add_argument("--output", default="results/candidate.png")
    visualize.add_argument("--history")
    visualize.add_argument("--title")
    visualize.set_defaults(function=command_visualize)

    gpu_benchmark = subparsers.add_parser("gpu-benchmark")
    gpu_benchmark.add_argument("--n", type=int, default=75)
    gpu_benchmark.add_argument("--population", type=int, default=1024)
    gpu_benchmark.add_argument("--repeats", type=int, default=5)
    gpu_benchmark.add_argument("--seed", type=int, default=91001)
    gpu_benchmark.set_defaults(function=command_gpu_benchmark)

    hardware = subparsers.add_parser("hardware")
    hardware.set_defaults(function=command_hardware)

    precompute_benchmark = subparsers.add_parser("precompute-benchmark")
    precompute_benchmark.add_argument("candidate")
    precompute_benchmark.add_argument("--n", type=int)
    precompute_benchmark.add_argument("--output")
    precompute_benchmark.set_defaults(function=command_precompute_benchmark)

    gpu_delta_benchmark = subparsers.add_parser("gpu-delta-benchmark")
    gpu_delta_benchmark.add_argument("candidate")
    gpu_delta_benchmark.add_argument("--n", type=int, default=75)
    gpu_delta_benchmark.add_argument("--anchors", type=int, default=5)
    gpu_delta_benchmark.add_argument("--repeats", type=int, default=100)
    gpu_delta_benchmark.set_defaults(function=command_gpu_delta_benchmark)

    gpu_search = subparsers.add_parser("gpu-search")
    gpu_search.add_argument("--n", type=int, default=75)
    gpu_search.add_argument("--initial", required=True)
    gpu_search.add_argument("--seed", type=int, required=True)
    gpu_search.add_argument("--generations", type=int, default=200)
    gpu_search.add_argument("--population", type=int, default=32768)
    gpu_search.add_argument("--elite", type=int, default=64)
    gpu_search.add_argument("--max-strength", type=int, default=12)
    gpu_search.add_argument("--energy-window", type=int, default=12)
    gpu_search.add_argument("--checkpoint-every", type=int, default=25)
    gpu_search.add_argument("--resume-population")
    gpu_search.add_argument("--additional-initial", nargs="*", default=[])
    gpu_search.add_argument("--additional-initial-dir", action="append", default=[])
    gpu_search.add_argument("--crossover-probability", type=float, default=0.0)
    gpu_search.add_argument("--engine", choices=("host", "device", "beam"), default="host")
    gpu_search.add_argument("--report-every", type=int, default=5)
    gpu_search.add_argument("--bad-bias", type=int, default=85)
    gpu_search.add_argument("--global-every", type=int, default=25)
    gpu_search.add_argument("--output", default="data/best/generated/gpu")
    gpu_search.set_defaults(function=command_gpu_search)

    gpu_four_cycle = subparsers.add_parser("gpu-four-cycle")
    gpu_four_cycle.add_argument("candidate")
    gpu_four_cycle.add_argument("--n", type=int, default=75)
    gpu_four_cycle.add_argument("--anchors", type=int, default=5)
    gpu_four_cycle.add_argument("--anchor-offset", type=int, default=0)
    gpu_four_cycle.add_argument("--rounds", type=int, default=10)
    gpu_four_cycle.add_argument("--batch-size", type=int, default=32768)
    gpu_four_cycle.add_argument("--output", default="data/best/generated/gpu-four-cycle")
    gpu_four_cycle.set_defaults(function=command_gpu_four_cycle)

    gpu_double_switch = subparsers.add_parser("gpu-double-switch")
    gpu_double_switch.add_argument("candidate")
    gpu_double_switch.add_argument("--n", type=int, default=75)
    gpu_double_switch.add_argument("--anchors", type=int, default=5)
    gpu_double_switch.add_argument("--anchor-offset", type=int, default=0)
    gpu_double_switch.add_argument("--batch-size", type=int, default=32768)
    gpu_double_switch.add_argument("--output", default="data/best/generated/gpu-double-switch")
    gpu_double_switch.set_defaults(function=command_gpu_double_switch)

    cp_sat = subparsers.add_parser("cp-sat")
    cp_sat.add_argument("--n", type=int, required=True)
    cp_sat.add_argument("--time-limit", type=float, default=60.0)
    cp_sat.add_argument("--seed", type=int, default=1)
    cp_sat.add_argument("--workers", type=int, default=8)
    cp_sat.add_argument("--log-search", action="store_true")
    cp_sat.add_argument("--hint")
    cp_sat.add_argument("--output", default="results/cp-sat-witness.json")
    cp_sat.set_defaults(function=command_cp_sat)

    perm_cp_sat = subparsers.add_parser("perm-cp-sat")
    perm_cp_sat.add_argument("--n", type=int, required=True)
    perm_cp_sat.add_argument("--time-limit", type=float, default=60.0)
    perm_cp_sat.add_argument("--seed", type=int, default=1)
    perm_cp_sat.add_argument("--workers", type=int, default=8)
    perm_cp_sat.add_argument("--log-search", action="store_true")
    perm_cp_sat.add_argument("--hint")
    perm_cp_sat.add_argument("--output", default="results/permutation-cp-sat-witness.json")
    perm_cp_sat.set_defaults(function=command_perm_cp_sat)

    rct4_sat = subparsers.add_parser("rct4-sat")
    rct4_sat.add_argument("--n", type=int, required=True)
    rct4_sat.add_argument("--time-limit", type=float, default=60.0)
    rct4_sat.add_argument("--solver", default="glucose42")
    rct4_sat.add_argument("--encoding", choices=("seqcounter", "direct", "totalizer", "cardnetwrk"), default="seqcounter")
    rct4_sat.add_argument("--hint")
    rct4_sat.add_argument("--force-hint", action="store_true")
    rct4_sat.add_argument("--output", default="results/rct4-sat-witness.json")
    rct4_sat.set_defaults(function=command_rct4_sat)

    seeds = subparsers.add_parser("seeds")
    seeds.add_argument("--source", required=True)
    seeds.add_argument("--target", type=int, default=75)
    seeds.add_argument("--keep", type=int, default=100)
    seeds.add_argument("--no-d4", action="store_true")
    seeds.add_argument("--single-decomposition", action="store_true")
    seeds.add_argument("--output", default="data/seeds/generated/n76-projection")
    seeds.set_defaults(function=command_seeds)

    expand_seeds = subparsers.add_parser("expand-seeds")
    expand_seeds.add_argument("--source", required=True)
    expand_seeds.add_argument("--target", type=int, default=75)
    expand_seeds.add_argument("--keep", type=int, default=100)
    expand_seeds.add_argument("--no-d4", action="store_true")
    expand_seeds.add_argument("--single-decomposition", action="store_true")
    expand_seeds.add_argument("--output", default="data/seeds/generated/n74-expansion")
    expand_seeds.set_defaults(function=command_expand_seeds)

    sample_expand = subparsers.add_parser("sample-expand")
    sample_expand.add_argument("--source", required=True)
    sample_expand.add_argument("--target", type=int, default=75)
    sample_expand.add_argument("--samples", type=int, default=100000)
    sample_expand.add_argument("--keep", type=int, default=100)
    sample_expand.add_argument("--seed", type=int, required=True)
    sample_expand.add_argument("--batch-size", type=int, default=8192)
    sample_expand.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    sample_expand.add_argument("--interior", action="store_true")
    sample_expand.add_argument("--output", default="data/seeds/generated/multistep-expansion")
    sample_expand.set_defaults(function=command_sample_expand)

    search = subparsers.add_parser("search")
    search.add_argument("--n", type=int, default=75)
    search.add_argument("--initial")
    search.add_argument("--seed", type=int, required=True)
    search.add_argument("--iterations", type=int, default=100000)
    search.add_argument("--tabu", type=int, default=11)
    search.add_argument("--plateau", type=int, default=800)
    search.add_argument("--output", default="data/best/generated")
    search.add_argument("--backend", choices=("cpu", "cuda-delta"), default="cpu")
    search.set_defaults(function=command_search)

    breakout = subparsers.add_parser("breakout")
    breakout.add_argument("--n", type=int, default=75)
    breakout.add_argument("--initial", required=True)
    breakout.add_argument("--seed", type=int, required=True)
    breakout.add_argument("--iterations", type=int, default=100000)
    breakout.add_argument("--plateau", type=int, default=500)
    breakout.add_argument("--weight-increment", type=int, default=1)
    breakout.add_argument("--max-weighted-lines", type=int, default=128)
    breakout.add_argument("--backend", choices=("cpu", "cuda-delta"), default="cuda-delta")
    breakout.add_argument("--output", default="data/best/generated/breakout")
    breakout.set_defaults(function=command_breakout)

    repair = subparsers.add_parser("repair")
    repair.add_argument("candidate")
    repair.add_argument("--n", type=int, default=75)
    repair.add_argument("--free-rows", type=int, default=16)
    repair.add_argument("--time-limit", type=float, default=30.0)
    repair.add_argument("--seed", type=int, default=1)
    repair.add_argument("--workers", type=int, default=1)
    repair.add_argument("--log-search", action="store_true")
    repair.add_argument("--output", default="results/n75-repaired.json")
    repair.set_defaults(function=command_repair)

    lazy_repair = subparsers.add_parser("lazy-repair")
    lazy_repair.add_argument("candidate")
    lazy_repair.add_argument("--n", type=int, default=75)
    lazy_repair.add_argument("--free-rows", type=int, default=40)
    lazy_repair.add_argument("--rounds", type=int, default=100)
    lazy_repair.add_argument("--time-limit", type=float, default=5.0)
    lazy_repair.add_argument("--seed", type=int, default=1)
    lazy_repair.add_argument("--workers", type=int, default=2)
    lazy_repair.add_argument("--row-strategy", choices=("all-conflicts", "cover-random"), default="cover-random")
    lazy_repair.add_argument("--no-hamming", action="store_true")
    lazy_repair.add_argument("--output", default="data/runs/lazy-repair")
    lazy_repair.set_defaults(function=command_lazy_repair)

    edge_repair = subparsers.add_parser("edge-repair")
    edge_repair.add_argument("candidate")
    edge_repair.add_argument("--n", type=int, default=75)
    edge_repair.add_argument("--free-points", type=int, default=50)
    edge_repair.add_argument("--time-limit", type=float, default=60.0)
    edge_repair.add_argument("--seed", type=int, default=1)
    edge_repair.add_argument("--workers", type=int, default=2)
    edge_repair.add_argument("--log-search", action="store_true")
    edge_repair.add_argument("--output", default="results/n75-edge-repaired.json")
    edge_repair.set_defaults(function=command_edge_repair)

    sat_repair = subparsers.add_parser("sat-repair")
    sat_repair.add_argument("candidate")
    sat_repair.add_argument("--n", type=int, default=75)
    sat_repair.add_argument("--free-rows", type=int, default=16)
    sat_repair.add_argument("--time-limit", type=float, default=30.0)
    sat_repair.add_argument("--seed", type=int, default=1)
    sat_repair.add_argument("--solver", default="glucose42")
    sat_repair.add_argument("--encoding", choices=("seqcounter", "direct", "totalizer", "cardnetwrk"), default="seqcounter")
    sat_repair.add_argument("--output", default="results/n75-sat-repaired.json")
    sat_repair.set_defaults(function=command_sat_repair)

    sat_distance = subparsers.add_parser("sat-distance")
    sat_distance.add_argument("candidate")
    sat_distance.add_argument("--n", type=int, default=75)
    sat_distance.add_argument("--max-removed", type=int, required=True)
    sat_distance.add_argument("--time-limit", type=float, default=120.0)
    sat_distance.add_argument("--solver", default="glucose42")
    sat_distance.add_argument("--encoding", choices=("seqcounter", "direct", "totalizer", "cardnetwrk"), default="seqcounter")
    sat_distance.add_argument("--output", default="results/n75-distance-witness.json")
    sat_distance.set_defaults(function=command_sat_distance)

    lns_portfolio = subparsers.add_parser("lns-portfolio")
    lns_portfolio.add_argument("candidate")
    lns_portfolio.add_argument("--n", type=int, default=75)
    lns_portfolio.add_argument("--backends", nargs="+", choices=("cp-sat", "sat"), default=["cp-sat", "sat"])
    lns_portfolio.add_argument("--free-rows", nargs="+", type=int, default=[24, 32, 40])
    lns_portfolio.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    lns_portfolio.add_argument("--time-limit", type=float, default=60.0)
    lns_portfolio.add_argument("--solver-workers", type=int, default=2)
    lns_portfolio.add_argument("--parallel", type=int, default=4)
    lns_portfolio.add_argument("--row-strategy", choices=("all-conflicts", "cover-random"), default="all-conflicts")
    lns_portfolio.add_argument("--output", default="data/runs/lns-portfolio")
    lns_portfolio.set_defaults(function=command_lns_portfolio)

    sat_assumptions = subparsers.add_parser("sat-assumptions")
    sat_assumptions.add_argument("candidate")
    sat_assumptions.add_argument("--n", type=int, default=75)
    sat_assumptions.add_argument("--free-rows", nargs="+", type=int, default=[40, 44, 48])
    sat_assumptions.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    sat_assumptions.add_argument("--time-limit", type=float, default=30.0)
    sat_assumptions.add_argument("--solver", default="glucose42")
    sat_assumptions.add_argument("--encoding", choices=("seqcounter", "direct", "totalizer", "cardnetwrk"), default="seqcounter")
    sat_assumptions.add_argument("--row-strategy", choices=("all-conflicts", "cover-random"), default="all-conflicts")
    sat_assumptions.add_argument("--output", default="data/runs/sat-assumptions")
    sat_assumptions.set_defaults(function=command_sat_assumptions)

    sym_search = subparsers.add_parser("sym-search")
    sym_search.add_argument("--n", type=int, default=75)
    sym_search.add_argument("--rounds", type=int, default=1000)
    sym_search.add_argument("--time-limit", type=float, default=10.0)
    sym_search.add_argument("--seed", type=int, default=1)
    sym_search.add_argument("--workers", type=int, default=8)
    sym_search.add_argument("--hint")
    sym_search.add_argument("--minimize-hint-distance", action="store_true")
    sym_search.add_argument("--output", default="data/best/generated/symmetry")
    sym_search.set_defaults(function=command_sym_search)

    rot2_search = subparsers.add_parser("rot2-search")
    rot2_search.add_argument("--n", type=int, default=75)
    rot2_search.add_argument("--rounds", type=int, default=1000)
    rot2_search.add_argument("--time-limit", type=float, default=10.0)
    rot2_search.add_argument("--seed", type=int, default=1)
    rot2_search.add_argument("--workers", type=int, default=8)
    rot2_search.add_argument("--output", default="data/best/generated/rot2")
    rot2_search.set_defaults(function=command_rot2_search)

    sym_full = subparsers.add_parser("sym-full")
    sym_full.add_argument("--n", type=int, default=75)
    sym_full.add_argument("--time-limit", type=float, default=600.0)
    sym_full.add_argument("--seed", type=int, default=1)
    sym_full.add_argument("--workers", type=int, default=8)
    sym_full.add_argument("--hint")
    sym_full.add_argument("--log-search", action="store_true")
    sym_full.add_argument("--output", default="results/symmetry-witness.json")
    sym_full.set_defaults(function=command_sym_full)

    rot2_full = subparsers.add_parser("rot2-full")
    rot2_full.add_argument("--n", type=int, default=75)
    rot2_full.add_argument("--time-limit", type=float, default=600.0)
    rot2_full.add_argument("--seed", type=int, default=1)
    rot2_full.add_argument("--workers", type=int, default=8)
    rot2_full.add_argument("--hint")
    rot2_full.add_argument("--log-search", action="store_true")
    rot2_full.add_argument("--output", default="results/rot2-witness.json")
    rot2_full.set_defaults(function=command_rot2_full)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.function(args))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
