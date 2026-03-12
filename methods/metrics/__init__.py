"""Shared metric entrypoints used by methods and experiments."""

from .prediction import align_prediction_targets, compute_prediction_metrics

__all__ = [
    "align_prediction_targets",
    "compute_prediction_metrics",
]
