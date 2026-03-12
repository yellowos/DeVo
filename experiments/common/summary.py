"""Scanning and CSV summarization helpers for experiment run results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .run import RESULT_FILENAME, read_run_result
from .schema import ExperimentRunResult
from .utils import flatten_mapping, to_csv_scalar


_BASE_SUMMARY_COLUMNS = [
    "schema_version",
    "experiment_name",
    "dataset",
    "method",
    "seed",
    "run_id",
    "status.state",
    "status.message",
    "status.error_type",
    "start_time",
    "end_time",
    "duration_seconds",
    "result_path",
    "run_dir",
]


def scan_run_results(results_root: str | Path) -> list[ExperimentRunResult]:
    """Recursively load every experiments-layer `result.json` file under a root."""

    root = Path(results_root).expanduser().resolve()
    if not root.exists():
        return []
    records: list[ExperimentRunResult] = []
    for result_path in sorted(root.rglob(RESULT_FILENAME)):
        records.append(read_run_result(result_path))
    return records


def build_run_summary_rows(
    records: Sequence[ExperimentRunResult],
) -> list[dict[str, Any]]:
    """Flatten per-run results into one-row-per-run dictionaries."""

    rows: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            item.experiment_name,
            item.dataset,
            item.method,
            -1 if item.seed is None else item.seed,
            item.run_id,
        ),
    ):
        row: dict[str, Any] = {
            "schema_version": record.schema_version,
            "experiment_name": record.experiment_name,
            "dataset": record.dataset,
            "method": record.method,
            "seed": record.seed,
            "run_id": record.run_id,
            "status.state": record.status.state,
            "status.message": record.status.message,
            "status.error_type": record.status.error_type,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "duration_seconds": record.duration_seconds,
        }
        row.update(
            flatten_mapping(
                {name: metric.to_dict() for name, metric in record.metrics.items()},
                prefix="metrics",
            )
        )
        row.update(
            flatten_mapping(
                {name: artifact.to_dict() for name, artifact in record.artifact_paths.items()},
                prefix="artifact_paths",
            )
        )
        row.update(flatten_mapping({"config": record.config}))
        row.update(flatten_mapping({"status": record.status.to_dict()}))
        row.update(flatten_mapping({"metadata": record.metadata}))
        rows.append(row)
    return rows


def _result_paths_for_rows(
    results_root: str | Path,
) -> dict[tuple[str, str, str, Any, str], tuple[str, str]]:
    root = Path(results_root).expanduser().resolve()
    path_index: dict[tuple[str, str, str, Any, str], tuple[str, str]] = {}
    for result_path in sorted(root.rglob(RESULT_FILENAME)):
        record = read_run_result(result_path)
        key = (
            record.experiment_name,
            record.dataset,
            record.method,
            record.seed,
            record.run_id,
        )
        path_index[key] = (str(result_path), str(result_path.parent))
    return path_index


def summarize_to_csv(
    records: Sequence[ExperimentRunResult] | Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    results_root: str | Path | None = None,
) -> Path:
    """Write a flat summary CSV that downstream collect/plot scripts can reuse."""

    if isinstance(records, Sequence) and records and isinstance(records[0], ExperimentRunResult):
        normalized_rows = build_run_summary_rows(records)
        if results_root is not None:
            path_index = _result_paths_for_rows(results_root)
            for row in normalized_rows:
                key = (
                    row["experiment_name"],
                    row["dataset"],
                    row["method"],
                    row["seed"],
                    row["run_id"],
                )
                result_path, run_dir = path_index.get(key, (None, None))
                row["result_path"] = result_path
                row["run_dir"] = run_dir
    elif isinstance(records, Sequence):
        normalized_rows = [dict(row) for row in records]
    else:
        normalized_rows = [dict(row) for row in records]

    for row in normalized_rows:
        row.setdefault("result_path", None)
        row.setdefault("run_dir", None)

    fieldnames = list(_BASE_SUMMARY_COLUMNS)
    extra_columns = sorted({key for row in normalized_rows for key in row.keys() if key not in fieldnames})
    fieldnames.extend(extra_columns)

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow({name: to_csv_scalar(row.get(name)) for name in fieldnames})
    return target
