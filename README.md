# No-three-in-line: geometry tools and search prototypes

Research source for placing two points in every row and column of an integer grid while avoiding collinear triples. The source includes determinant and normalized-slope verifiers, permutation-based CPU search, and optional SAT/CP-SAT/GPU experiments.

## Included candidate

`results/best-near-solution-E10.json` stores 150 integer-coordinate points for n=75. It is a saved **near-solution**, with exactly ten collinear triples verified by independent integer arithmetic. It must be rejected by both exact verifiers. It is a diagnostic/regression fixture, not a valid solution or a proof of nonexistence.

This publication contains one coordinate set and omits external configuration tables, run directories, solver logs, large search histories and benchmark/proof summaries. It makes no claim about today's global research frontier or novelty.

## Verification scripts

These scripts use Python 3.11+ and the standard library:

```powershell
python tools/verify_solution.py results/best-near-solution-E10.json --n 75
python tools/verify_solution_alt.py results/best-near-solution-E10.json --n 75
```

Both commands are expected to print `INVALID` and exit with status 1 for this near-solution. A valid integer witness would need to pass both independent algorithms.

The JSON file interfaces now require a positive integer grid size and arrays of exactly two genuine integer coordinates. Bool, float, numeric-string and malformed point aliases are rejected instead of silently coerced. The standalone verifier keeps an independent schema guard. Valid explicit grid overrides and zero-based coordinates are preserved.

This strict guarantee applies to JSON. The retained legacy plaintext/compact interfaces have separate parsing rules; in particular the old free-form plaintext integer-extraction grammar is permissive. Use JSON for untrusted candidate verification.

## Finite CPU checks

With pytest and NumPy already available, run from this repository:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -B -m pytest -p no:cacheprovider tests/test_geometry.py tests/test_verifier_agreement.py tests/test_state.py tests/test_delta.py tests/test_analysis.py
```

The selected tests use synthetic states and check integer geometry, agreement between independent algorithms, state invariants and incremental scores against full recomputation. They do not need external configuration records, model/provider services or a GPU. Numba is an optional CPU accelerator; its JIT cache, when present, belongs to the local working copy.

The complete `no3` CLI imports solver modules at startup. Its SAT/CP-SAT paths need the optional solver dependencies listed in pyproject.toml. GPU/visualization paths have separate optional dependencies and are not required for the verifier scripts or the selected CPU tests. Commands expecting omitted data/known-solutions or run files need caller-supplied records.

## Attribution and licensing

See ATTRIBUTION.txt for the published formulations and configuration sources referenced during development. External source/configuration collections are not bundled in this snapshot. No standalone project license was present in the inspected local source; license selection remains explicit owner work.

Historical compute counters, local UNSAT neighborhoods and dated literature searches are intentionally outside the claims of this minimal release. Publication and static inspection alone do not establish runtime correctness; record the actual check outcomes when preparing a release.


Actual publication-copy results: [VALIDATION.md](VALIDATION.md). No original source, search history or runtime data was modified to prepare this copy.
