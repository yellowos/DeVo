"""Small shared utilities for the experiments layer."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def filesystem_token(value: str, *, default: str = "unknown") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return normalized or default


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return to_jsonable(value.tolist())
        except TypeError:
            pass
    return str(value)


def to_csv_scalar(value: Any) -> Any:
    jsonable = to_jsonable(value)
    if jsonable is None or isinstance(jsonable, (bool, int, float, str)):
        return jsonable
    return json.dumps(jsonable, sort_keys=True, ensure_ascii=True)


def flatten_mapping(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_mapping(value, prefix=next_key))
        else:
            flat[next_key] = to_csv_scalar(value)
    return flat
