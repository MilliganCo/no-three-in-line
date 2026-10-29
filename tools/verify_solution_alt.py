#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from no_three_75.io import load_candidate
from no_three_75.verifier_b import verify


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent normalized-slope verifier")
    parser.add_argument("candidate")
    parser.add_argument("--n", type=int)
    args = parser.parse_args()
    try:
        n, points, _ = load_candidate(args.candidate, n=args.n)
        valid, detail = verify(points, n)
    except (OSError, ValueError, TypeError) as exc:
        print(f"INVALID: {exc}")
        return 1
    print(f"{'VALID' if valid else 'INVALID'} n={n} points={len(points)}: {detail}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
