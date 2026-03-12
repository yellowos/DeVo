"""Collect experiments-layer run results into a flat CSV table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.common import DEFAULT_RESULTS_ROOT, scan_run_results, summarize_to_csv
else:
    from .common import DEFAULT_RESULTS_ROOT, scan_run_results, summarize_to_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Recommended entry: python -m experiments.collect <results_root> [--output <csv>].",
    )
    parser.add_argument(
        "results_root",
        nargs="?",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Directory containing experiment run subdirectories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to <results_root>/summary.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results_root = Path(args.results_root).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else results_root / "summary.csv"
    )

    records = scan_run_results(results_root)
    summarize_to_csv(records, output_path, results_root=results_root)
    print(f"Wrote {len(records)} run rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
