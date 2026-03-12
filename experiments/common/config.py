"""Config loading helpers for the experiments layer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from .utils import compute_mapping_hash, stable_json_dumps

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ should provide tomllib.
    tomllib = None


ConfigSource: TypeAlias = Mapping[str, Any] | str | Path | None


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {str(key): copy.deepcopy(value) for key, value in base.items()}
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_config_file(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML config loading requires Python 3.11+ or tomllib.")
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    else:
        raise ValueError(f"Unsupported config suffix '{path.suffix}' for {path}.")

    if not isinstance(payload, Mapping):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    return payload


def load_experiment_config(
    source: ConfigSource,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a config mapping from a dict, JSON file, or TOML file."""

    if source is None:
        config: Mapping[str, Any] = {}
    elif isinstance(source, Mapping):
        config = dict(source)
    else:
        path = Path(source).expanduser().resolve()
        config = _read_config_file(path)

    return finalize_config(config, overrides=overrides)


def deep_merge_configs(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _deep_merge(base or {}, override or {})


def finalize_config(
    config: Mapping[str, Any] | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return copy.deepcopy(deep_merge_configs(config, overrides))


def compute_config_hash(config: Mapping[str, Any]) -> str:
    return compute_mapping_hash(config)


def write_resolved_config(
    config: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(stable_json_dumps(config))
        handle.write("\n")
    return target
