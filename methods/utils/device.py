"""Centralized PyTorch device and dtype policy for methods layer."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is absent.
    torch = None


_ALLOWED_DEVICE_TYPES = {"auto", "mps", "cpu"}
_KNOWN_DTYPES = {
    "float16",
    "float32",
    "float64",
    "bfloat16",
}
_MPS_SAFE_DTYPES = {"float16", "float32"}
_CPU_DEFAULT_DTYPE = "float32"
_MPS_DEFAULT_DTYPE = "float32"


def is_torch_available() -> bool:
    return torch is not None


def is_mps_available() -> bool:
    if torch is None:
        return False
    if not hasattr(torch.backends, "mps"):
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _normalize_device_name(preferred_device: Optional[str]) -> str:
    if preferred_device is None:
        return "auto"
    normalized = str(preferred_device).strip().lower()
    if normalized not in _ALLOWED_DEVICE_TYPES:
        return "auto"
    return normalized


def _normalize_dtype_name(preferred_dtype: Any) -> Optional[str]:
    if preferred_dtype is None:
        return None
    if torch is not None and isinstance(preferred_dtype, torch.dtype):
        for dtype_name in _KNOWN_DTYPES:
            if getattr(torch, dtype_name) == preferred_dtype:
                return dtype_name
        return None
    normalized = str(preferred_dtype).strip().lower()
    return normalized if normalized in _KNOWN_DTYPES else None


@dataclass(frozen=True)
class DeviceContext:
    """Resolved device policy shared by all methods."""

    device_type: str
    dtype_name: str
    device: Any
    dtype: Any
    torch_available: bool
    mps_available: bool
    fallback_reason: Optional[str] = None
    platform_name: str = platform.system()

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "dtype": self.dtype_name,
            "torch_available": self.torch_available,
            "mps_available": self.mps_available,
            "fallback_reason": self.fallback_reason,
            "platform_name": self.platform_name,
        }


def resolve_dtype(
    preferred_dtype: Any = None,
    *,
    device_type: Optional[str] = None,
) -> tuple[str, Any, Optional[str]]:
    """Resolve a safe dtype for the requested device."""

    normalized_device = _normalize_device_name(device_type)
    normalized_dtype = _normalize_dtype_name(preferred_dtype)

    if normalized_device == "mps":
        if normalized_dtype in _MPS_SAFE_DTYPES:
            dtype_name = normalized_dtype
            reason = None
        elif normalized_dtype is None:
            dtype_name = _MPS_DEFAULT_DTYPE
            reason = None
        else:
            dtype_name = _MPS_DEFAULT_DTYPE
            reason = f"MPS does not safely support dtype='{normalized_dtype}', falling back to '{dtype_name}'."
    else:
        if normalized_dtype is None:
            dtype_name = _CPU_DEFAULT_DTYPE
            reason = None
        else:
            dtype_name = normalized_dtype
            reason = None

    torch_dtype = getattr(torch, dtype_name) if torch is not None else dtype_name
    return dtype_name, torch_dtype, reason


def select_device(
    *,
    preferred_device: Optional[str] = None,
    preferred_dtype: Any = None,
    allow_mps: bool = True,
) -> DeviceContext:
    """Resolve the runtime device.

    Auto policy is intentionally simple and deterministic:
    1. Prefer MPS on Apple Silicon when available.
    2. Otherwise fall back to CPU.
    3. Do not auto-select CUDA.
    """

    requested_device = _normalize_device_name(preferred_device)
    mps_available = is_mps_available()
    fallback_reasons: list[str] = []

    if requested_device == "mps":
        if allow_mps and mps_available:
            device_type = "mps"
        else:
            device_type = "cpu"
            fallback_reasons.append("Requested MPS, but MPS is unavailable. Falling back to CPU.")
    elif requested_device == "cpu":
        device_type = "cpu"
    else:
        device_type = "mps" if allow_mps and mps_available else "cpu"

    dtype_name, torch_dtype, dtype_reason = resolve_dtype(
        preferred_dtype=preferred_dtype,
        device_type=device_type,
    )
    if dtype_reason:
        fallback_reasons.append(dtype_reason)

    if torch is not None:
        device = torch.device(device_type)
    else:
        device = device_type
        fallback_reasons.append("PyTorch is not installed; returning string-based device context.")

    return DeviceContext(
        device_type=device_type,
        dtype_name=dtype_name,
        device=device,
        dtype=torch_dtype,
        torch_available=is_torch_available(),
        mps_available=mps_available,
        fallback_reason=" ".join(fallback_reasons) if fallback_reasons else None,
    )
