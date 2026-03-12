"""Polynomial NARMAX-style baseline for the methods layer."""

from __future__ import annotations

import csv
import json
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional, Sequence

import numpy as np

from methods.base import BaseMethod, KernelRecoveryResult, MethodResult, register_method


PathLike = str | Path
_EPS = 1e-12


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return array


def _coerce_window_array(value: Any, *, name: str) -> np.ndarray:
    array = _as_float_array(value, name=name)
    if array.ndim == 1:
        array = array[:, np.newaxis, np.newaxis]
    elif array.ndim == 2:
        array = array[:, :, np.newaxis]
    elif array.ndim >= 4:
        array = array.reshape(array.shape[0], array.shape[1], -1)
    return array


def _coerce_target_array(value: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    array = _as_float_array(value, name="Y")
    if array.ndim == 1:
        array = array[:, np.newaxis]
    sample_shape = tuple(int(dim) for dim in array.shape[1:])
    flattened = array.reshape(array.shape[0], -1)
    return flattened, sample_shape


def _coerce_history_array(
    value: Any,
    *,
    name: str,
    num_samples: int,
) -> np.ndarray:
    array = _coerce_window_array(value, name=name)
    if array.shape[0] != num_samples:
        raise ValueError(f"{name} sample count mismatch: expected {num_samples}, got {array.shape[0]}.")
    return array


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _sample_indices(num_samples: int, max_samples: Optional[int]) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or num_samples <= max_samples:
        return np.arange(num_samples, dtype=np.int64)
    return np.linspace(0, num_samples - 1, num=max_samples, dtype=np.int64)


def _safe_column_scale(matrix: np.ndarray) -> np.ndarray:
    scale = matrix.std(axis=0, ddof=0)
    scale = np.where(scale < _EPS, 1.0, scale)
    return scale


def _factor_signature(base_spec: Mapping[str, Any], power: int) -> dict[str, Any]:
    payload = {
        "base_index": int(base_spec["base_index"]),
        "source_kind": str(base_spec["source_kind"]),
        "channel": int(base_spec["channel"]),
        "lag": int(base_spec["lag"]),
        "label": str(base_spec["label"]),
        "power": int(power),
    }
    return payload


def _term_group(factors: Sequence[Mapping[str, Any]]) -> str:
    kinds = {str(item["source_kind"]) for item in factors}
    if not kinds:
        return "bias"
    if len(kinds) == 1:
        kind = next(iter(kinds))
        if kind == "input":
            return "input_polynomial"
        if kind == "output":
            return "autoregressive"
        if kind == "residual":
            return "moving_average"
    return "mixed"


@register_method("narmax")
class NARMAXMethod(BaseMethod):
    """Ridge-regularized polynomial regressor over lagged signals.

    The shared dataset bundle currently provides exogenous input windows by default,
    so this implementation behaves as a NARMAX-style baseline with optional explicit
    output/residual histories supplied through `fit(..., output_history=..., residual_history=...)`
    and `predict(..., output_history=..., residual_history=...)`.
    """

    SUPPORTS_KERNEL_RECOVERY: ClassVar[bool] = True
    DEFAULT_CONFIG: ClassVar[dict[str, Any]] = {
        "input_lags": 8,
        "output_lags": 0,
        "moving_average_lags": 0,
        "polynomial_order": 2,
        "include_bias": True,
        "max_base_terms": 12,
        "max_terms": 256,
        "ranking_sample_size": 20000,
        "ridge_alpha": 1e-6,
        "fit_split": "train",
    }

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        merged_config = dict(self.DEFAULT_CONFIG)
        merged_config.update(dict(config or {}))
        super().__init__(config=merged_config, device=device, dtype=dtype)
        self.coefficients_: Optional[np.ndarray] = None
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_scale_: Optional[np.ndarray] = None
        self.base_regressors_: list[dict[str, Any]] = []
        self.term_specs_: list[dict[str, Any]] = []
        self.target_sample_shape_: tuple[int, ...] = ()
        self.target_descriptors_: list[dict[str, Any]] = []
        self.window_shape_: dict[str, int] = {}
        self.recovery_summary_: dict[str, Any] = {}

    def fit(
        self,
        dataset_bundle: Any,
        *,
        split: Optional[str] = None,
        output_history: Any = None,
        residual_history: Any = None,
    ) -> MethodResult:
        bundle = self.normalize_dataset_bundle(dataset_bundle)
        split_name = str(split or self.config.get("fit_split", "train"))
        fit_split = bundle.get_split(split_name)

        X = _coerce_window_array(fit_split.X, name="X")
        Y_flat, target_sample_shape = _coerce_target_array(fit_split.Y)

        input_lags = min(int(self.config["input_lags"]), int(X.shape[1]))
        output_lags = int(self.config["output_lags"])
        moving_average_lags = int(self.config["moving_average_lags"])
        polynomial_order = max(1, int(self.config["polynomial_order"]))
        include_bias = bool(self.config["include_bias"])
        max_base_terms = max(1, int(self.config["max_base_terms"]))
        max_terms = max(1, int(self.config["max_terms"]))
        ranking_sample_size = int(self.config["ranking_sample_size"])
        ridge_alpha = float(self.config["ridge_alpha"])

        output_array = None
        if output_lags > 0 and output_history is not None:
            output_array = _coerce_history_array(
                output_history,
                name="output_history",
                num_samples=X.shape[0],
            )

        residual_array = None
        if moving_average_lags > 0 and residual_history is not None:
            residual_array = _coerce_history_array(
                residual_history,
                name="residual_history",
                num_samples=X.shape[0],
            )

        candidate_matrix, candidate_specs = self._build_candidate_regressors(
            X,
            output_history=output_array,
            residual_history=residual_array,
            input_lags=input_lags,
            output_lags=output_lags,
            moving_average_lags=moving_average_lags,
        )
        selected_base_indices = self._select_base_regressors(
            candidate_matrix,
            Y_flat,
            max_terms=max_base_terms,
            max_samples=ranking_sample_size,
        )
        self.base_regressors_ = [
            {**candidate_specs[index], "base_index": position}
            for position, index in enumerate(selected_base_indices)
        ]
        base_matrix = candidate_matrix[:, selected_base_indices]

        monomial_groups = self._select_monomial_groups(
            base_matrix,
            Y_flat,
            polynomial_order=polynomial_order,
            max_terms=max_terms - (1 if include_bias else 0),
            max_samples=ranking_sample_size,
        )
        self.term_specs_ = self._build_term_specs(monomial_groups, include_bias=include_bias)
        design_matrix = self._build_design_matrix(base_matrix, self.term_specs_)

        feature_mean = design_matrix.mean(axis=0)
        feature_scale = _safe_column_scale(design_matrix)
        if include_bias and self.term_specs_ and self.term_specs_[0]["group"] == "bias":
            feature_mean[0] = 0.0
            feature_scale[0] = 1.0
        normalized_design = (design_matrix - feature_mean) / feature_scale
        if include_bias and self.term_specs_ and self.term_specs_[0]["group"] == "bias":
            normalized_design[:, 0] = 1.0

        gram = normalized_design.T @ normalized_design
        regularizer = ridge_alpha * np.eye(gram.shape[0], dtype=np.float64)
        if include_bias and self.term_specs_ and self.term_specs_[0]["group"] == "bias":
            regularizer[0, 0] = 0.0
        rhs = normalized_design.T @ Y_flat
        try:
            coefficients = np.linalg.solve(gram + regularizer, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(gram + regularizer) @ rhs

        self.coefficients_ = coefficients
        self.feature_mean_ = feature_mean
        self.feature_scale_ = feature_scale
        self.target_sample_shape_ = target_sample_shape
        self.target_descriptors_ = self._build_target_descriptors(target_sample_shape)
        self.window_shape_ = {
            "window_length": int(X.shape[1]),
            "input_channels": int(X.shape[2]),
            "output_history_channels": int(output_array.shape[2]) if output_array is not None else 0,
            "residual_history_channels": int(residual_array.shape[2]) if residual_array is not None else 0,
        }
        self.recovery_summary_ = {
            "representation_type": "narmax_structural_coefficients",
            "exact_kernel_recovery": False,
            "uses_explicit_output_history": bool(output_array is not None and output_lags > 0),
            "uses_explicit_residual_history": bool(residual_array is not None and moving_average_lags > 0),
            "input_only_bundle_compatible": True,
            "term_basis": "normalized_polynomial_regressors",
        }
        self.training_summary = {
            "dataset_name": bundle.meta.dataset_name,
            "fit_split": split_name,
            "train_samples": int(bundle.train.num_samples),
            "val_samples": int(bundle.val.num_samples),
            "test_samples": int(bundle.test.num_samples),
            "window_length": int(X.shape[1]),
            "input_channels": int(X.shape[2]),
            "selected_base_terms": int(base_matrix.shape[1]),
            "selected_structure_terms": int(len(self.term_specs_)),
            "effective_input_lags": int(input_lags),
            "configured_output_lags": int(output_lags),
            "effective_output_lags": int(output_array.shape[1]) if output_array is not None else 0,
            "configured_moving_average_lags": int(moving_average_lags),
            "effective_moving_average_lags": int(residual_array.shape[1]) if residual_array is not None else 0,
            "polynomial_order": int(polynomial_order),
            "ridge_alpha": float(ridge_alpha),
            "target_shape": list(target_sample_shape),
            "supports_partial_kernel_recovery": True,
        }
        self.is_fitted = True
        return MethodResult(
            predictions=None,
            model_state_path=self.model_state_path,
            training_summary=dict(self.training_summary),
            artifacts={},
            metadata={
                "supports_kernel_recovery": True,
                "recovery_level": "structural_coefficients_only",
            },
        )

    def predict(
        self,
        X: Any,
        *,
        output_history: Any = None,
        residual_history: Any = None,
    ) -> np.ndarray:
        if not self.is_fitted or self.coefficients_ is None:
            raise RuntimeError("NARMAXMethod must be fitted before predict().")
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("Missing feature normalization state.")

        input_array = _coerce_window_array(X, name="X")
        output_array = None
        if int(self.config["output_lags"]) > 0 and output_history is not None:
            output_array = _coerce_history_array(
                output_history,
                name="output_history",
                num_samples=input_array.shape[0],
            )
        residual_array = None
        if int(self.config["moving_average_lags"]) > 0 and residual_history is not None:
            residual_array = _coerce_history_array(
                residual_history,
                name="residual_history",
                num_samples=input_array.shape[0],
            )

        base_matrix = self._build_base_matrix_from_specs(
            input_array,
            output_history=output_array,
            residual_history=residual_array,
        )
        design_matrix = self._build_design_matrix(base_matrix, self.term_specs_)
        normalized_design = (design_matrix - self.feature_mean_) / self.feature_scale_
        if self.term_specs_ and self.term_specs_[0]["group"] == "bias":
            normalized_design[:, 0] = 1.0
        predictions = normalized_design @ self.coefficients_
        if not self.target_sample_shape_:
            return predictions
        return predictions.reshape((predictions.shape[0],) + self.target_sample_shape_)

    def export_artifacts(self, output_dir: PathLike) -> Mapping[str, Any]:
        if not self.is_fitted or self.coefficients_ is None:
            raise RuntimeError("NARMAXMethod must be fitted before export_artifacts().")

        target_dir = Path(output_dir).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)

        structure_path = target_dir / "narmax_structure.json"
        summary_path = target_dir / "narmax_summary.json"
        csv_path = target_dir / "narmax_coefficients.csv"
        coefficients_path = target_dir / "narmax_coefficients.npy"

        recovery = self.recover_kernels()
        _json_dump(
            structure_path,
            {
                "kernels": recovery.kernels,
                "summary": recovery.summary,
            },
        )
        _json_dump(
            summary_path,
            {
                "method_name": self.method_name,
                "config": dict(self.config),
                "training_summary": dict(self.training_summary),
                "window_shape": dict(self.window_shape_),
            },
        )
        np.save(coefficients_path, self.coefficients_)
        self._write_coefficients_csv(csv_path)

        return {
            "structure_json": str(structure_path),
            "summary_json": str(summary_path),
            "coefficients_csv": str(csv_path),
            "coefficients_npy": str(coefficients_path),
        }

    def recover_kernels(
        self,
        *,
        coefficient_threshold: float = 0.0,
        include_zero_terms: bool = False,
    ) -> KernelRecoveryResult:
        if not self.is_fitted or self.coefficients_ is None:
            raise RuntimeError("NARMAXMethod must be fitted before recover_kernels().")
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("Missing feature normalization state.")

        grouped_terms: dict[str, list[dict[str, Any]]] = {
            "input_polynomial_terms": [],
            "autoregressive_terms": [],
            "moving_average_terms": [],
            "mixed_terms": [],
            "bias_terms": [],
        }
        kept_terms = 0
        for term_index, spec in enumerate(self.term_specs_):
            coefficient_vector = self.coefficients_[term_index]
            max_magnitude = float(np.max(np.abs(coefficient_vector)))
            if not include_zero_terms and max_magnitude <= coefficient_threshold:
                continue
            kept_terms += 1
            payload = {
                "term_index": int(term_index),
                "label": str(spec["label"]),
                "degree": int(spec["degree"]),
                "group": str(spec["group"]),
                "factors": [dict(factor) for factor in spec["factors"]],
                "coefficient": coefficient_vector.tolist(),
                "feature_mean": float(self.feature_mean_[term_index]),
                "feature_scale": float(self.feature_scale_[term_index]),
            }
            if spec["group"] == "bias":
                grouped_terms["bias_terms"].append(payload)
            elif spec["group"] == "input_polynomial":
                grouped_terms["input_polynomial_terms"].append(payload)
            elif spec["group"] == "autoregressive":
                grouped_terms["autoregressive_terms"].append(payload)
            elif spec["group"] == "moving_average":
                grouped_terms["moving_average_terms"].append(payload)
            else:
                grouped_terms["mixed_terms"].append(payload)

        summary = {
            **dict(self.recovery_summary_),
            "selected_terms": int(len(self.term_specs_)),
            "returned_terms": int(kept_terms),
            "coefficient_threshold": float(coefficient_threshold),
            "targets": len(self.target_descriptors_),
        }
        kernels = {
            "representation_type": "narmax_structural_coefficients",
            "term_basis": self.recovery_summary_.get("term_basis", "normalized_polynomial_regressors"),
            "exact_volterra_kernels": False,
            "targets": [dict(descriptor) for descriptor in self.target_descriptors_],
            "window_shape": dict(self.window_shape_),
            **grouped_terms,
        }
        return KernelRecoveryResult(
            kernels=kernels,
            summary=summary,
            artifacts={},
        )

    def get_state(self) -> Mapping[str, Any]:
        return {
            "coefficients": self.coefficients_.tolist() if self.coefficients_ is not None else None,
            "feature_mean": self.feature_mean_.tolist() if self.feature_mean_ is not None else None,
            "feature_scale": self.feature_scale_.tolist() if self.feature_scale_ is not None else None,
            "base_regressors": [dict(spec) for spec in self.base_regressors_],
            "term_specs": [self._serialize_term_spec(spec) for spec in self.term_specs_],
            "target_sample_shape": list(self.target_sample_shape_),
            "target_descriptors": [dict(item) for item in self.target_descriptors_],
            "window_shape": dict(self.window_shape_),
            "recovery_summary": dict(self.recovery_summary_),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        coefficients = state.get("coefficients")
        self.coefficients_ = None if coefficients is None else np.asarray(coefficients, dtype=np.float64)
        feature_mean = state.get("feature_mean")
        self.feature_mean_ = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float64)
        feature_scale = state.get("feature_scale")
        self.feature_scale_ = None if feature_scale is None else np.asarray(feature_scale, dtype=np.float64)
        self.base_regressors_ = [dict(spec) for spec in state.get("base_regressors", [])]
        self.term_specs_ = [self._deserialize_term_spec(spec) for spec in state.get("term_specs", [])]
        self.target_sample_shape_ = tuple(int(value) for value in state.get("target_sample_shape", []))
        self.target_descriptors_ = [dict(item) for item in state.get("target_descriptors", [])]
        self.window_shape_ = {
            str(key): int(value)
            for key, value in dict(state.get("window_shape", {})).items()
        }
        self.recovery_summary_ = dict(state.get("recovery_summary", {}))

    def _build_candidate_regressors(
        self,
        X: np.ndarray,
        *,
        output_history: Optional[np.ndarray],
        residual_history: Optional[np.ndarray],
        input_lags: int,
        output_lags: int,
        moving_average_lags: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        columns: list[np.ndarray] = []
        specs: list[dict[str, Any]] = []

        for lag in range(1, input_lags + 1):
            position = X.shape[1] - lag
            for channel in range(X.shape[2]):
                columns.append(X[:, position, channel])
                specs.append(
                    {
                        "source_kind": "input",
                        "channel": int(channel),
                        "lag": int(lag),
                        "label": f"input[c{channel},lag_{lag}]",
                    }
                )

        if output_history is not None and output_lags > 0:
            effective_lags = min(output_lags, int(output_history.shape[1]))
            for lag in range(1, effective_lags + 1):
                position = output_history.shape[1] - lag
                for channel in range(output_history.shape[2]):
                    columns.append(output_history[:, position, channel])
                    specs.append(
                        {
                            "source_kind": "output",
                            "channel": int(channel),
                            "lag": int(lag),
                            "label": f"output[c{channel},lag_{lag}]",
                        }
                    )

        if residual_history is not None and moving_average_lags > 0:
            effective_lags = min(moving_average_lags, int(residual_history.shape[1]))
            for lag in range(1, effective_lags + 1):
                position = residual_history.shape[1] - lag
                for channel in range(residual_history.shape[2]):
                    columns.append(residual_history[:, position, channel])
                    specs.append(
                        {
                            "source_kind": "residual",
                            "channel": int(channel),
                            "lag": int(lag),
                            "label": f"residual[c{channel},lag_{lag}]",
                        }
                    )

        if not columns:
            raise ValueError("No candidate regressors were constructed. Check lag configuration.")
        matrix = np.column_stack(columns).astype(np.float64, copy=False)
        return matrix, specs

    def _select_base_regressors(
        self,
        candidate_matrix: np.ndarray,
        targets: np.ndarray,
        *,
        max_terms: int,
        max_samples: Optional[int],
    ) -> np.ndarray:
        if candidate_matrix.shape[1] <= max_terms:
            return np.arange(candidate_matrix.shape[1], dtype=np.int64)

        row_indices = _sample_indices(candidate_matrix.shape[0], max_samples)
        candidate_sample = candidate_matrix[row_indices]
        target_sample = targets[row_indices]
        candidate_centered = candidate_sample - candidate_sample.mean(axis=0, keepdims=True)
        target_centered = target_sample - target_sample.mean(axis=0, keepdims=True)
        candidate_norm = np.linalg.norm(candidate_centered, axis=0)
        target_norm = np.linalg.norm(target_centered, axis=0)
        candidate_norm = np.where(candidate_norm < _EPS, 1.0, candidate_norm)
        target_norm = np.where(target_norm < _EPS, 1.0, target_norm)
        correlations = np.abs(candidate_centered.T @ target_centered) / np.outer(candidate_norm, target_norm)
        scores = correlations.max(axis=1)
        selected = np.argsort(-scores)[:max_terms]
        return np.sort(selected.astype(np.int64))

    def _select_monomial_groups(
        self,
        base_matrix: np.ndarray,
        targets: np.ndarray,
        *,
        polynomial_order: int,
        max_terms: int,
        max_samples: Optional[int],
    ) -> list[tuple[int, ...]]:
        groups: list[tuple[int, ...]] = []
        for degree in range(1, polynomial_order + 1):
            groups.extend(combinations_with_replacement(range(base_matrix.shape[1]), degree))

        if len(groups) <= max_terms:
            return groups

        row_indices = _sample_indices(base_matrix.shape[0], max_samples)
        base_sample = base_matrix[row_indices]
        target_sample = targets[row_indices]
        target_centered = target_sample - target_sample.mean(axis=0, keepdims=True)
        target_norm = np.linalg.norm(target_centered, axis=0)
        target_norm = np.where(target_norm < _EPS, 1.0, target_norm)

        scores = np.zeros(len(groups), dtype=np.float64)
        for index, group in enumerate(groups):
            term = np.prod(base_sample[:, group], axis=1)
            term_centered = term - term.mean()
            term_norm = float(np.linalg.norm(term_centered))
            if term_norm < _EPS:
                continue
            scores[index] = float(np.max(np.abs(term_centered @ target_centered) / (term_norm * target_norm)))
        selected = np.argsort(-scores)[:max_terms]
        selected_groups = [groups[index] for index in np.sort(selected)]
        return selected_groups

    def _build_term_specs(
        self,
        groups: Sequence[Sequence[int]],
        *,
        include_bias: bool,
    ) -> list[dict[str, Any]]:
        term_specs: list[dict[str, Any]] = []
        if include_bias:
            term_specs.append(
                {
                    "group": "bias",
                    "degree": 0,
                    "base_indices": [],
                    "factors": [],
                    "label": "bias",
                }
            )

        for group in groups:
            counts: dict[int, int] = {}
            for base_index in group:
                counts[int(base_index)] = counts.get(int(base_index), 0) + 1
            factors = [_factor_signature(self.base_regressors_[index], power) for index, power in sorted(counts.items())]
            label_parts = []
            for factor in factors:
                if factor["power"] == 1:
                    label_parts.append(str(factor["label"]))
                else:
                    label_parts.append(f"{factor['label']}^{factor['power']}")
            term_specs.append(
                {
                    "group": _term_group(factors),
                    "degree": int(len(group)),
                    "base_indices": [int(value) for value in group],
                    "factors": factors,
                    "label": " * ".join(label_parts),
                }
            )
        return term_specs

    def _build_design_matrix(
        self,
        base_matrix: np.ndarray,
        term_specs: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        design = np.empty((base_matrix.shape[0], len(term_specs)), dtype=np.float64)
        for index, spec in enumerate(term_specs):
            base_indices = list(spec["base_indices"])
            if not base_indices:
                design[:, index] = 1.0
                continue
            design[:, index] = np.prod(base_matrix[:, base_indices], axis=1)
        return design

    def _build_base_matrix_from_specs(
        self,
        X: np.ndarray,
        *,
        output_history: Optional[np.ndarray],
        residual_history: Optional[np.ndarray],
    ) -> np.ndarray:
        columns: list[np.ndarray] = []
        for spec in self.base_regressors_:
            source_kind = spec["source_kind"]
            lag = int(spec["lag"])
            channel = int(spec["channel"])
            if source_kind == "input":
                source = X
            elif source_kind == "output":
                if output_history is None:
                    raise ValueError("predict() requires output_history for configured autoregressive terms.")
                source = output_history
            elif source_kind == "residual":
                if residual_history is None:
                    raise ValueError("predict() requires residual_history for configured moving-average terms.")
                source = residual_history
            else:
                raise ValueError(f"Unknown source kind: {source_kind}")

            if lag > int(source.shape[1]):
                raise ValueError(
                    f"{source_kind} history too short for lag {lag}: only {source.shape[1]} steps available."
                )
            if channel >= int(source.shape[2]):
                raise ValueError(
                    f"{source_kind} history missing channel {channel}: only {source.shape[2]} channels available."
                )
            position = source.shape[1] - lag
            columns.append(source[:, position, channel])
        return np.column_stack(columns).astype(np.float64, copy=False)

    def _build_target_descriptors(self, target_shape: tuple[int, ...]) -> list[dict[str, Any]]:
        if not target_shape:
            return [{"flat_index": 0}]
        if len(target_shape) == 1:
            return [
                {
                    "flat_index": int(index),
                    "output_index": int(index),
                }
                for index in range(target_shape[0])
            ]
        if len(target_shape) == 2:
            descriptors: list[dict[str, Any]] = []
            flat_index = 0
            for horizon_index in range(target_shape[0]):
                for output_channel in range(target_shape[1]):
                    descriptors.append(
                        {
                            "flat_index": int(flat_index),
                            "horizon_index": int(horizon_index),
                            "output_channel": int(output_channel),
                        }
                    )
                    flat_index += 1
            return descriptors
        return [
            {
                "flat_index": int(index),
                "indices": list(np.unravel_index(index, target_shape)),
            }
            for index in range(int(np.prod(target_shape)))
        ]

    def _write_coefficients_csv(self, path: Path) -> None:
        if self.coefficients_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("Missing fitted state for coefficient export.")

        fieldnames = [
            "term_index",
            "label",
            "group",
            "degree",
            "feature_mean",
            "feature_scale",
            "coefficients",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, spec in enumerate(self.term_specs_):
                writer.writerow(
                    {
                        "term_index": int(index),
                        "label": spec["label"],
                        "group": spec["group"],
                        "degree": int(spec["degree"]),
                        "feature_mean": float(self.feature_mean_[index]),
                        "feature_scale": float(self.feature_scale_[index]),
                        "coefficients": json.dumps(self.coefficients_[index].tolist()),
                    }
                )

    def _serialize_term_spec(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "group": str(spec["group"]),
            "degree": int(spec["degree"]),
            "base_indices": [int(value) for value in spec["base_indices"]],
            "factors": [dict(item) for item in spec["factors"]],
            "label": str(spec["label"]),
        }

    def _deserialize_term_spec(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "group": str(spec["group"]),
            "degree": int(spec["degree"]),
            "base_indices": [int(value) for value in spec.get("base_indices", [])],
            "factors": [dict(item) for item in spec.get("factors", [])],
            "label": str(spec["label"]),
        }
