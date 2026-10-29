# Publication verification — 2026-10-29

PASS — all 60 tests included in this source snapshot. The 45 new strict-JSON cases first produced
39 failures / 6 passes on the unmodified parser, then all 45 passed after the guard changes.
The other 15 synthetic CPU tests cover geometry/state/delta-score agreement.

PASS — independent direct integer calculation: n=75, 150 unique in-range points, two in every
row/column, exactly **10 collinear triples**. Both package verifiers reject this near-solution;
the synthetic n=2 square passes. All three verifier CLIs reject the retained near-solution with
the expected INVALID output and exit 1. This expected rejection is not a failed release gate.

The two geometric algorithms, solver/search/state modules and compact alphabet are unchanged.
Only input schema/domain guards were hardened. JSON coercion is fixed; the legacy free-form
plaintext grammar remains permissive and is not a strict untrusted-input interface.

NOT RUN — new search/solver experiments, GPU, external data sets or literature/novelty verification.
This is a research toolkit and an invalid diagnostic fixture, not a valid n=75 solution or global proof.
External configuration archives, private Git history, runs and large proof/benchmark summaries omitted.
