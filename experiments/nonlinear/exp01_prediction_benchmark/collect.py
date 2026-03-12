from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import compute_config_hash


DEFAULT_DATASET_ORDER = [
    "duffing",
    "silverbox",
    "volterra_wiener",
    "coupled_duffing",
    "cascaded_tanks",
]
DEFAULT_METHOD_ORDER = [
    "narmax",
    "tt_volterra",
    "cp_volterra",
    "laguerre_volterra",
    "mlp",
    "lstm",
    "devo",
]


def _resolve_path(base_dir: Path, raw: str | None, *, fallback: str) -> Path:
    target = Path(raw or fallback)
    if not target.is_absolute():
        target = (base_dir / target).resolve()
    return target


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return payload


def _resolve_profile(config_path: Path, profile: str) -> dict[str, Any]:
    raw = _load_yaml(config_path)
    profiles = dict(raw.get("profiles", {}) or {})
    if profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown profile '{profile}'. Available: {available}")
    profile_payload = dict(profiles[profile] or {})
    config_dir = config_path.parent.resolve()
    return {
        "datasets": list(profile_payload.get("datasets", DEFAULT_DATASET_ORDER)),
        "methods": list(profile_payload.get("methods", DEFAULT_METHOD_ORDER)),
        "results_dir": _resolve_path(
            config_dir,
            str(profile_payload.get("output_dir", "")) or None,
            fallback=f"./outputs/{profile}",
        ),
    }


def _read_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Result payload at {path} must be an object.")
    payload["result_path"] = str(path.resolve())
    return payload


def _write_runs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "seed",
        "run_id",
        "config_hash",
        "status",
        "nmse",
        "rmse",
        "mse",
        "result_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = dict(row.get("metrics", {}) or {})
            writer.writerow(
                {
                    "dataset": row.get("dataset"),
                    "method": row.get("method"),
                    "seed": row.get("seed"),
                    "run_id": row.get("run_id"),
                    "config_hash": row.get("config_hash"),
                    "status": row.get("status"),
                    "nmse": metrics.get("nmse"),
                    "rmse": metrics.get("rmse"),
                    "mse": metrics.get("mse"),
                    "result_path": row.get("result_path"),
                }
            )


def _is_valid_run_dir(run_dir: Path) -> bool:
    required = (
        run_dir / "result.json",
        run_dir / "status.json",
        run_dir / "run_context.json",
        run_dir / "resolved_config.json",
        run_dir / "metrics.json",
        run_dir / "artifacts_manifest.json",
    )
    if any(not path.is_file() for path in required):
        return False
    try:
        result_payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        config_payload = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(result_payload, dict) or not isinstance(status_payload, dict) or not isinstance(config_payload, dict):
        return False
    if result_payload.get("run_id") != status_payload.get("run_id"):
        return False
    return result_payload.get("config_hash") == compute_config_hash(config_payload)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    abs_value = abs(value)
    if abs_value != 0.0 and (abs_value >= 1_000 or abs_value < 1e-4):
        return f"{value:.3e}"
    return f"{value:.6f}"


def _format_mean_std(mean: float | None, std: float | None, success_count: int) -> str:
    if mean is None:
        return f"- (n={success_count})"
    return f"{_format_metric(mean)} +/- {_format_metric(std or 0.0)} (n={success_count})"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float], mean: float | None) -> float | None:
    if not values or mean is None:
        return None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _aggregate(results: list[dict[str, Any]], dataset_order: list[str], method_order: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        dataset = str(result.get("dataset"))
        method = str(result.get("method"))
        key = (dataset, method)
        grouped.setdefault(
            key,
            {
                "dataset": dataset,
                "method": method,
                "successful_nmse": [],
                "successful_rmse": [],
                "num_runs": 0,
                "num_success": 0,
                "num_failed": 0,
                "num_skipped": 0,
            },
        )
        group = grouped[key]
        group["num_runs"] += 1
        status = str(result.get("status", "unknown"))
        if status == "success":
            metrics = dict(result.get("metrics", {}) or {})
            nmse = _safe_float(metrics.get("nmse"))
            rmse = _safe_float(metrics.get("rmse"))
            if nmse is not None:
                group["successful_nmse"].append(nmse)
            if rmse is not None:
                group["successful_rmse"].append(rmse)
            group["num_success"] += 1
        elif status == "skipped":
            group["num_skipped"] += 1
        else:
            group["num_failed"] += 1

    summary_rows: list[dict[str, Any]] = []
    ordered_keys = [(dataset, method) for dataset in dataset_order for method in method_order]
    ordered_keys.extend(sorted(key for key in grouped if key not in ordered_keys))

    for key in ordered_keys:
        row = grouped.get(
            key,
            {
                "dataset": key[0],
                "method": key[1],
                "successful_nmse": [],
                "successful_rmse": [],
                "num_runs": 0,
                "num_success": 0,
                "num_failed": 0,
                "num_skipped": 0,
            },
        )
        nmse_mean = _mean(row["successful_nmse"])
        nmse_std = _std(row["successful_nmse"], nmse_mean)
        rmse_mean = _mean(row["successful_rmse"])
        rmse_std = _std(row["successful_rmse"], rmse_mean)
        summary_rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "num_runs": row["num_runs"],
                "num_success": row["num_success"],
                "num_failed": row["num_failed"],
                "num_skipped": row["num_skipped"],
                "nmse_mean": nmse_mean,
                "nmse_std": nmse_std,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
            }
        )
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "num_runs",
        "num_success",
        "num_failed",
        "num_skipped",
        "nmse_mean",
        "nmse_std",
        "rmse_mean",
        "rmse_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_metric_matrix(
    rows: list[dict[str, Any]],
    *,
    datasets: list[str],
    methods: list[str],
    metric_name: str,
) -> list[list[str]]:
    row_lookup = {(row["method"], row["dataset"]): row for row in rows}
    matrix: list[list[str]] = []
    for method in methods:
        cells = [method]
        for dataset in datasets:
            row = row_lookup.get((method, dataset))
            if row is None:
                cells.append("-")
                continue
            cells.append(
                _format_mean_std(
                    _safe_float(row.get(f"{metric_name}_mean")),
                    _safe_float(row.get(f"{metric_name}_std")),
                    int(row.get("num_success", 0)),
                )
            )
        matrix.append(cells)
    return matrix


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _latex_table(caption: str, label: str, headers: list[str], rows: list[list[str]]) -> str:
    column_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        " & ".join(_latex_escape(cell) for cell in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(cell) for cell in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect nonlinear benchmark experiment 01 results.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the experiment YAML config.",
    )
    parser.add_argument("--profile", default="full", help="Config profile to read.")
    parser.add_argument("--results-dir", type=Path, help="Optional results directory override.")
    parser.add_argument("--output-dir", type=Path, help="Optional summary output directory override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = _resolve_profile(args.config.resolve(), args.profile)
    results_dir = (args.results_dir or profile["results_dir"]).expanduser().resolve()
    summary_dir = (args.output_dir or results_dir / "summary").expanduser().resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(
        run_dir / "result.json" for run_dir in results_dir.rglob("*") if _is_valid_run_dir(run_dir)
    )
    if not result_files:
        raise FileNotFoundError(f"No result.json files found under {results_dir}")

    results = [_read_result(path) for path in result_files]
    summary_rows = _aggregate(results, profile["datasets"], profile["methods"])
    if not summary_rows:
        raise RuntimeError(f"No aggregated rows could be built from {results_dir}")

    long_csv_path = summary_dir / "benchmark_summary_long.csv"
    _write_csv(long_csv_path, summary_rows)
    runs_csv_path = summary_dir / "benchmark_runs.csv"
    _write_runs_csv(runs_csv_path, results)

    headers = ["method", *profile["datasets"]]
    nmse_rows = _build_metric_matrix(summary_rows, datasets=profile["datasets"], methods=profile["methods"], metric_name="nmse")
    rmse_rows = _build_metric_matrix(summary_rows, datasets=profile["datasets"], methods=profile["methods"], metric_name="rmse")

    nmse_csv_path = summary_dir / "benchmark_nmse_table.csv"
    with nmse_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(nmse_rows)

    rmse_csv_path = summary_dir / "benchmark_rmse_table.csv"
    with rmse_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rmse_rows)

    markdown = "\n\n".join(
        [
            "# Nonlinear Benchmark Prediction Summary",
            "## NMSE",
            _markdown_table(headers, nmse_rows),
            "## RMSE",
            _markdown_table(headers, rmse_rows),
        ]
    )
    (summary_dir / "benchmark_tables.md").write_text(markdown + "\n", encoding="utf-8")

    latex = "\n\n".join(
        [
            _latex_table(
                "Prediction benchmark NMSE (mean +/- std over seeds).",
                "tab:exp01_nmse",
                headers,
                nmse_rows,
            ),
            _latex_table(
                "Prediction benchmark RMSE (mean +/- std over seeds).",
                "tab:exp01_rmse",
                headers,
                rmse_rows,
            ),
        ]
    )
    (summary_dir / "benchmark_tables.tex").write_text(latex + "\n", encoding="utf-8")

    status_payload = {
        "results_dir": str(results_dir),
        "summary_dir": str(summary_dir),
        "num_result_files": len(result_files),
        "num_summary_rows": len(summary_rows),
        "long_csv": str(long_csv_path),
        "runs_csv": str(runs_csv_path),
        "nmse_csv": str(nmse_csv_path),
        "rmse_csv": str(rmse_csv_path),
    }
    (summary_dir / "collect_manifest.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    print(json.dumps(status_payload, indent=2))


if __name__ == "__main__":
    main()
