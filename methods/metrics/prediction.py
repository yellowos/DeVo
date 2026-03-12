"""Prediction metrics with explicit domain metadata."""

from __future__ import annotations

from typing import Any

import numpy as np


def align_prediction_targets(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true_array = np.asarray(y_true, dtype=np.float64)
    pred_array = np.asarray(y_pred, dtype=np.float64)

    if true_array.shape[0] != pred_array.shape[0]:
        raise ValueError(
            f"Prediction batch mismatch: expected {true_array.shape[0]} samples, got {pred_array.shape[0]}."
        )

    if true_array.shape == pred_array.shape:
        return true_array, pred_array

    true_flat = true_array.reshape(true_array.shape[0], -1)
    pred_flat = pred_array.reshape(pred_array.shape[0], -1)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(
            f"Prediction shape mismatch: expected flattened shape {true_flat.shape}, got {pred_flat.shape}."
        )
    return true_flat, pred_flat


def compute_prediction_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    domain: str = "native",
) -> dict[str, Any]:
    true_array, pred_array = align_prediction_targets(y_true, y_pred)
    error = pred_array - true_array
    mse = float(np.mean(np.square(error)))
    rmse = float(np.sqrt(mse))

    true_flat = true_array.reshape(true_array.shape[0], -1)
    centered = true_flat - true_flat.mean(axis=0, keepdims=True)
    variance = float(np.mean(np.square(centered)))
    if variance <= 1e-12:
        signal_power = float(np.mean(np.square(true_flat)))
        variance = signal_power if signal_power > 1e-12 else float("nan")
    nmse = float(mse / variance) if np.isfinite(variance) else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "nmse": nmse,
        "domain": domain,
    }
