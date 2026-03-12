"""CLI entrypoint for experiments-layer result summarization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common import collect_experiment_summary, write_summary_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect experiments-layer result summaries.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Repository root. Defaults to the parent of this script.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="Root directory that contains experiment result folders. Defaults to <project-root>/experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated summary outputs. Defaults to <experiments-root>/_summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_experiment_summary(
        project_root=args.project_root,
        experiments_root=args.experiments_root,
        output_dir=args.output_dir,
    )
    outputs = write_summary_outputs(report, output_dir=args.output_dir)

    print(f"Wrote experiments summary to {report.output_dir}")
    for label, path in outputs.items():
        print(f"- {label}: {path}")
    print("Status counts:")
    for status, count in sorted(report.status_counts.items(), key=lambda item: item[0]):
        print(f"  {status}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
