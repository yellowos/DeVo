"""Run-directory helpers for experiments-layer execution and persistence."""

from __future__ import annotations

import json
import socket
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .schema import (
    ArtifactPathRef,
    ExperimentRunResult,
    RunStatusRecord,
    artifact_paths_from_method_result,
    coerce_artifact_paths,
    coerce_metrics,
)
from .utils import filesystem_token, utc_now_iso


RESULT_FILENAME = "result.json"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def _default_run_id() -> str:
    timestamp = utc_now_iso().replace(":", "").replace("-", "")
    return f"run_{timestamp}_{uuid.uuid4().hex[:8]}"


def _seed_token(seed: Optional[int]) -> str:
    return "seed_none" if seed is None else f"seed_{seed}"


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
    run_id: str


def _build_paths(
    results_root: str | Path,
    *,
    experiment_name: str,
    dataset: str,
    method: str,
    seed: Optional[int],
    run_id: Optional[str] = None,
) -> ExperimentPaths:
    root = Path(results_root).expanduser().resolve()
    experiment_dir = root / filesystem_token(experiment_name)
    dataset_dir = experiment_dir / filesystem_token(dataset)
    method_dir = dataset_dir / filesystem_token(method)
    seed_dir = method_dir / _seed_token(seed)
    resolved_run_id = filesystem_token(run_id or _default_run_id(), default=_default_run_id())
    run_dir = seed_dir / resolved_run_id
    return ExperimentPaths(
        results_root=root,
        experiment_dir=experiment_dir,
        dataset_dir=dataset_dir,
        method_dir=method_dir,
        seed_dir=seed_dir,
        run_dir=run_dir,
        result_path=run_dir / RESULT_FILENAME,
        run_id=resolved_run_id,
    )


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
        duration_seconds = _duration_seconds(self.start_time, finish_time)

        result = ExperimentRunResult(
            run_id=self.paths.run_id,
            experiment_name=self.experiment_name,
            dataset=self.dataset,
            method=self.method,
            seed=self.seed,
            config=dict(self.config),
            metrics=coerce_metrics(metrics),
            artifact_paths=combined_artifacts,
            status=status,
            start_time=self.start_time,
            end_time=finish_time,
            duration_seconds=duration_seconds,
            metadata={
                "hostname": socket.gethostname(),
                "python_version": sys.version.split()[0],
                **dict(metadata or {}),
            },
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
) -> ExperimentRunHandle:
    paths = _build_paths(
        results_root,
        experiment_name=experiment_name,
        dataset=dataset,
        method=method,
        seed=seed,
        run_id=run_id,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    return ExperimentRunHandle(
        paths=paths,
        experiment_name=experiment_name,
        dataset=dataset,
        method=method,
        seed=seed,
        config=dict(config or {}),
        start_time=utc_now_iso(),
    )
