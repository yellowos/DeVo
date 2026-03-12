"""Collect TEP experiment 01 results into paper-ready summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime only
    raise RuntimeError("PyYAML is required to collect experiment results.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect five-unit fault isolation results.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profile", type=str, default="full")
    parser.add_argument("--results-root", type=Path, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping at {path}")
    return payload


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_results_root(config: Mapping[str, Any], *, config_path: Path, profile: str, override: Optional[Path]) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    paths = dict(config.get("paths", {}))
    output_root = resolve_path(config_path.parent.resolve(), paths["output_root"])
    return output_root / profile


def load_result_files(results_root: Path) -> List[Path]:
    return sorted(results_root.glob("h*/**/result.json"))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_mean(payload: Mapping[str, Any], key: str) -> Optional[float]:
    aggregate = payload.get("aggregate_metrics", {})
    metric = aggregate.get(key, {})
    value = metric.get("mean")
    return None if value is None else float(value)


def build_rows(result_paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for result_path in result_paths:
        payload = load_json(result_path)
        aggregate = payload.get("aggregate_metrics", {})
        row = {
            "method_id": payload.get("method_id"),
            "method": payload.get("display_name", payload.get("method_id")),
            "horizon": payload.get("horizon"),
            "status": payload.get("status"),
            "processed_runs": aggregate.get("processed_runs"),
            "evaluable_runs": aggregate.get("evaluable_runs"),
            "top1": metric_mean(payload, "top1"),
            "top3": metric_mean(payload, "top3"),
            "soft_precision_at_3": metric_mean(payload, "soft_precision_at_3"),
            "early_hit": metric_mean(payload, "early_hit"),
            "normal_test_mse": (payload.get("normal_test_summary") or {}).get("mse"),
            "skip_reason": payload.get("skip_reason"),
            "result_file": str(result_path),
        }
        rows.append(row)
    rows.sort(key=lambda row: (int(row["horizon"]), str(row["method"])))
    return rows


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Method",
        "h",
        "Status",
        "Processed",
        "Evaluable",
        "Top-1",
        "Top-3",
        "Soft P@3",
        "Early Hit",
        "Normal MSE",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["horizon"]),
                    str(row["status"]),
                    str(row["processed_runs"]),
                    str(row["evaluable_runs"]),
                    format_metric(row["top1"]),
                    format_metric(row["top3"]),
                    format_metric(row["soft_precision_at_3"]),
                    format_metric(row["early_hit"]),
                    format_metric(row["normal_test_mse"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    results_root = resolve_results_root(config, config_path=args.config.resolve(), profile=args.profile, override=args.results_root)
    result_paths = load_result_files(results_root)
    rows = build_rows(result_paths)

    collected_dir = results_root / "collected"
    collected_dir.mkdir(parents=True, exist_ok=True)

    table_payload = {
        "profile": args.profile,
        "results_root": str(results_root),
        "row_count": len(rows),
        "rows": rows,
    }
    with (collected_dir / "paper_table_5_3_3.json").open("w", encoding="utf-8") as handle:
        json.dump(table_payload, handle, indent=2, ensure_ascii=False)
    write_csv(collected_dir / "paper_table_5_3_3.csv", rows)
    markdown = markdown_table(rows)
    (collected_dir / "paper_table_5_3_3.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
