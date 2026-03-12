"""Run-directory helpers for experiments-layer execution and persistence."""

from __future__ import annotations

import json
import socket
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import compute_config_hash, finalize_config, write_resolved_config
from .schema import (
    ArtifactPathRef,
    ExperimentRunResult,
    RunStatusRecord,
    artifact_paths_from_method_result,
    coerce_artifact_paths,
    coerce_metrics,
)
from .utils import compute_mapping_hash, filesystem_token, to_jsonable, utc_now_iso


RESULT_FILENAME = "result.json"
STATUS_FILENAME = "status.json"
RUN_CONTEXT_FILENAME = "run_context.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.json"
METRICS_FILENAME = "metrics.json"
ARTIFACTS_MANIFEST_FILENAME = "artifacts_manifest.json"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def _seed_token(seed: Optional[int]) -> str:
    return "seed_none" if seed is None else f"seed_{seed}"


def make_run_id(
    *,
    experiment_name: str | None = None,
    dataset: str | None = None,
    method: str | None = None,
    seed: Optional[int] = None,
    started_at: Optional[str] = None,
) -> str:
    timestamp = (started_at or utc_now_iso()).replace("-", "").replace(":", "").replace("+00:00", "Z")
    return "_".join(
        [
            filesystem_token(experiment_name or "run"),
            filesystem_token(dataset or "dataset"),
            filesystem_token(method or "method"),
            _seed_token(seed),
            filesystem_token(timestamp, default="time"),
            uuid.uuid4().hex[:8],
        ]
    )


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem layout for one experiments-layer run."""

    results_root: Path
    experiment_dir: Path
    dataset_dir: Path
    method_dir: Path
    seed_dir: Path
    run_dir: Path
    result_path: Path
    status_path: Path
    run_context_path: Path
    resolved_config_path: Path
    metrics_path: Path
    artifacts_manifest_path: Path
    run_id: str


def _build_paths(
    results_root: str | Path,
    *,
    experiment_name: str,
    dataset: str,
    method: str,
    seed: Optional[int],
    run_id: str,
) -> ExperimentPaths:
    root = Path(results_root).expanduser().resolve()
    experiment_dir = root / filesystem_token(experiment_name)
    dataset_dir = experiment_dir / filesystem_token(dataset)
    method_dir = dataset_dir / filesystem_token(method)
    seed_dir = method_dir / _seed_token(seed)
    resolved_run_id = filesystem_token(run_id, default="run")
    run_dir = seed_dir / resolved_run_id
    return ExperimentPaths(
        results_root=root,
        experiment_dir=experiment_dir,
        dataset_dir=dataset_dir,
        method_dir=method_dir,
        seed_dir=seed_dir,
        run_dir=run_dir,
        result_path=run_dir / RESULT_FILENAME,
        status_path=run_dir / STATUS_FILENAME,
        run_context_path=run_dir / RUN_CONTEXT_FILENAME,
        resolved_config_path=run_dir / RESOLVED_CONFIG_FILENAME,
        metrics_path=run_dir / METRICS_FILENAME,
        artifacts_manifest_path=run_dir / ARTIFACTS_MANIFEST_FILENAME,
        run_id=resolved_run_id,
    )


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
    return target


def write_run_result(result: ExperimentRunResult, result_path: str | Path) -> Path:
    target = Path(result_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, sort_keys=True)
    return target


def read_run_result(result_path: str | Path) -> ExperimentRunResult:
    source = Path(result_path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Run result must be a JSON object: {source}")
    return ExperimentRunResult.from_mapping(payload)


def prepare_run_dir(
    *,
    results_root: str | Path,
    experiment_name: str,
    dataset: str,
    method: str,
    seed: Optional[int],
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
) -> ExperimentPaths:
    paths = _build_paths(
        results_root,
        experiment_name=experiment_name,
        dataset=dataset,
        method=method,
        seed=seed,
        run_id=run_id
        or make_run_id(
            experiment_name=experiment_name,
            dataset=dataset,
            method=method,
            seed=seed,
            started_at=started_at,
        ),
    )
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    return paths


def write_run_context(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return _write_json(path, payload)


def write_status(
    path: str | Path,
    *,
    status: RunStatusRecord,
    run_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    return _write_json(
        path,
        {
            "run_id": run_id,
            "updated_at": utc_now_iso(),
            **status.to_dict(),
            "metadata": dict(metadata or {}),
        },
    )


def compute_split_hash(payload: Mapping[str, Any]) -> str:
    return compute_mapping_hash(payload)


@dataclass
class ExperimentRunHandle:
    """Minimal helper to create a run directory and finalize one terminal status."""

    paths: ExperimentPaths
    experiment_name: str
    dataset: str
    method: str
    seed: Optional[int]
    config: dict[str, Any]
    start_time: str
    config_hash: str
    split_hash: Optional[str] = None

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "python_version": sys.version.split()[0],
            "config_hash": self.config_hash,
            "resolved_config_path": str(self.paths.resolved_config_path),
            "run_context_path": str(self.paths.run_context_path),
            "status_path": str(self.paths.status_path),
            "metrics_path": str(self.paths.metrics_path),
            "artifacts_manifest_path": str(self.paths.artifacts_manifest_path),
            "split_hash": self.split_hash,
        }

    def finalize(
        self,
        *,
        status: RunStatusRecord,
        metrics: Optional[Mapping[str, Any]] = None,
        artifact_paths: Optional[Mapping[str, Any]] = None,
        method_result: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        end_time: Optional[str] = None,
    ) -> Path:
        finish_time = end_time or utc_now_iso()
        combined_artifacts: dict[str, ArtifactPathRef] = {}
        combined_artifacts.update(artifact_paths_from_method_result(method_result))
        combined_artifacts.update(coerce_artifact_paths(artifact_paths))
        coerced_metrics = coerce_metrics(metrics)
        duration_seconds = _duration_seconds(self.start_time, finish_time)
        metadata_payload = {
            **self._base_metadata(),
            **dict(metadata or {}),
        }

        write_status(
            self.paths.status_path,
            status=status,
            run_id=self.paths.run_id,
            metadata=metadata_payload,
        )
        _write_json(
            self.paths.metrics_path,
            {
                "run_id": self.paths.run_id,
                "metrics": {name: metric.to_dict() for name, metric in coerced_metrics.items()},
            },
        )
        _write_json(
            self.paths.artifacts_manifest_path,
            {
                "run_id": self.paths.run_id,
                "artifacts": {name: artifact.to_dict() for name, artifact in combined_artifacts.items()},
            },
        )

        result = ExperimentRunResult(
            run_id=self.paths.run_id,
            experiment_name=self.experiment_name,
            dataset=self.dataset,
            method=self.method,
            seed=self.seed,
            config=finalize_config(self.config),
            metrics=coerced_metrics,
            artifact_paths=combined_artifacts,
            status=status,
            start_time=self.start_time,
            end_time=finish_time,
            duration_seconds=duration_seconds,
            metadata=metadata_payload,
        )
        return write_run_result(result, self.paths.result_path)

    def save_success(
        self,
        *,
        metrics: Optional[Mapping[str, Any]] = None,
        artifact_paths: Optional[Mapping[str, Any]] = None,
        method_result: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        message: str = "completed",
    ) -> Path:
        return self.finalize(
            status=RunStatusRecord.success(message),
            metrics=metrics,
            artifact_paths=artifact_paths,
            method_result=method_result,
            metadata=metadata,
        )

    def save_skipped(
        self,
        message: str,
        *,
        metrics: Optional[Mapping[str, Any]] = None,
        artifact_paths: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        return self.finalize(
            status=RunStatusRecord.skipped(message, metadata=metadata),
            metrics=metrics,
            artifact_paths=artifact_paths,
            metadata=metadata,
        )

    def save_failed(
        self,
        *,
        message: Optional[str] = None,
        exc: BaseException | None = None,
        metrics: Optional[Mapping[str, Any]] = None,
        artifact_paths: Optional[Mapping[str, Any]] = None,
        method_result: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        traceback_text: Optional[str] = None,
    ) -> Path:
        return self.finalize(
            status=RunStatusRecord.failed(
                message=message,
                exc=exc,
                traceback_text=traceback_text,
                metadata=metadata,
            ),
            metrics=metrics,
            artifact_paths=artifact_paths,
            method_result=method_result,
            metadata=metadata,
        )


def _duration_seconds(start_time: Optional[str], end_time: Optional[str]) -> Optional[float]:
    if not start_time or not end_time:
        return None
    try:
        start = _parse_iso_timestamp(start_time)
        end = _parse_iso_timestamp(end_time)
    except ValueError:
        return None
    return max(0.0, (end - start).total_seconds())


def _parse_iso_timestamp(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def create_run_handle(
    *,
    experiment_name: str,
    dataset: str,
    method: str,
    seed: Optional[int],
    config: Optional[Mapping[str, Any]] = None,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    run_id: Optional[str] = None,
    split_hash: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> ExperimentRunHandle:
    started_at = utc_now_iso()
    paths = prepare_run_dir(
        results_root=results_root,
        experiment_name=experiment_name,
        dataset=dataset,
        method=method,
        seed=seed,
        run_id=run_id,
        started_at=started_at,
    )
    resolved_config = finalize_config(config or {})
    config_hash = compute_config_hash(resolved_config)
    write_resolved_config(resolved_config, paths.resolved_config_path)

    run_context = {
        "run_id": paths.run_id,
        "experiment_name": experiment_name,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "start_time": started_at,
        "config_hash": config_hash,
        "resolved_config_path": str(paths.resolved_config_path),
        "split_hash": split_hash,
        **dict(context or {}),
    }
    write_run_context(paths.run_context_path, run_context)
    initial_status = RunStatusRecord.running("running")
    write_status(paths.status_path, status=initial_status, run_id=paths.run_id, metadata=run_context)
    _write_json(paths.metrics_path, {"run_id": paths.run_id, "metrics": {}})
    _write_json(paths.artifacts_manifest_path, {"run_id": paths.run_id, "artifacts": {}})
    write_run_result(
        ExperimentRunResult(
            run_id=paths.run_id,
            experiment_name=experiment_name,
            dataset=dataset,
            method=method,
            seed=seed,
            config=resolved_config,
            metrics={},
            artifact_paths={},
            status=initial_status,
            start_time=started_at,
            end_time=None,
            duration_seconds=None,
            metadata={
                "config_hash": config_hash,
                "resolved_config_path": str(paths.resolved_config_path),
                "run_context_path": str(paths.run_context_path),
                "split_hash": split_hash,
            },
        ),
        paths.result_path,
    )
    return ExperimentRunHandle(
        paths=paths,
        experiment_name=experiment_name,
        dataset=dataset,
        method=method,
        seed=seed,
        config=resolved_config,
        start_time=started_at,
        config_hash=config_hash,
        split_hash=split_hash,
    )


def is_valid_run_dir(path: str | Path) -> bool:
    run_dir = Path(path).expanduser().resolve()
    required = (
        run_dir / RESULT_FILENAME,
        run_dir / STATUS_FILENAME,
        run_dir / RESOLVED_CONFIG_FILENAME,
        run_dir / METRICS_FILENAME,
        run_dir / ARTIFACTS_MANIFEST_FILENAME,
    )
    if not run_dir.is_dir() or any(not item.is_file() for item in required):
        return False
    try:
        result = read_run_result(run_dir / RESULT_FILENAME)
        with (run_dir / STATUS_FILENAME).open("r", encoding="utf-8") as handle:
            status_payload = json.load(handle)
        with (run_dir / RESOLVED_CONFIG_FILENAME).open("r", encoding="utf-8") as handle:
            config_payload = json.load(handle)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    if not isinstance(status_payload, Mapping) or not isinstance(config_payload, Mapping):
        return False
    if status_payload.get("run_id") != result.run_id:
        return False
    return compute_config_hash(config_payload) == result.metadata.get("config_hash")


def iter_valid_run_dirs(results_root: str | Path) -> list[Path]:
    root = Path(results_root).expanduser().resolve()
    if not root.exists():
        return []
    run_dirs: list[Path] = []
    for result_path in sorted(root.rglob(RESULT_FILENAME)):
        run_dir = result_path.parent
        if is_valid_run_dir(run_dir):
            run_dirs.append(run_dir)
    return run_dirs
