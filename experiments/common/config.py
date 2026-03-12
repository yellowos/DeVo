"""Config loading helpers for the experiments layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TypeAlias

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ should provide tomllib.
    tomllib = None


ConfigSource: TypeAlias = Mapping[str, Any] | str | Path | None


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
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

    if overrides:
        return _deep_merge(config, overrides)
    return dict(config)
