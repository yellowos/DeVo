"""Lagged linear baselines used for ARX / VAR-style comparisons.

These methods export first-order linear coefficients, but they do not claim
kernel recovery under the methods-layer `recover_kernels()` protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from methods.base import ArtifactRef, BaseMethod, MethodResult, register_method


PathLike = str | Path

_RECOVERY_SEMANTICS = (
    "Exports first-order linear coefficients for inspection, but does not mark "
    "them as recovered kernels under BaseMethod.recover_kernels()."
)


def _safe_prod(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _ensure_2d_coef(coef: np.ndarray) -> np.ndarray:
    if coef.ndim == 1:
        return coef.reshape(1, -1)
    return coef


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return array


def _coerce_feature_batch(
    value: Any,
    *,
    expected_window_length: Optional[int] = None,
    expected_input_dim: Optional[int] = None,
    allow_single_sample: bool = False,
) -> np.ndarray:
    array = _as_float_array(value, name="X")

    if allow_single_sample and expected_window_length and expected_input_dim:
        if array.ndim == 1 and array.size == expected_window_length * expected_input_dim:
            return array.reshape(1, expected_window_length, expected_input_dim)
        if expected_input_dim == 1 and array.ndim == 1 and array.size == expected_window_length:
            return array.reshape(1, expected_window_length, 1)
        if array.ndim == 2 and array.shape == (expected_window_length, expected_input_dim):
            return array.reshape(1, expected_window_length, expected_input_dim)

    if array.ndim == 1:
        raise ValueError("X must include a batch dimension.")

    if array.ndim >= 3:
        return array.reshape(array.shape[0], array.shape[1], _safe_prod(tuple(array.shape[2:])))

    if expected_window_length and expected_input_dim and array.shape[1] == expected_window_length * expected_input_dim:
        return array.reshape(array.shape[0], expected_window_length, expected_input_dim)
    if expected_window_length and expected_input_dim == 1 and array.shape[1] == expected_window_length:
        return array.reshape(array.shape[0], expected_window_length, 1)
    if expected_window_length == 1 and expected_input_dim and array.shape[1] == expected_input_dim:
        return array.reshape(array.shape[0], 1, expected_input_dim)

    return array.reshape(array.shape[0], 1, array.shape[1])


def _coerce_target_batch(
    value: Any,
    *,
    expected_horizon: Optional[int] = None,
    expected_output_dim: Optional[int] = None,
) -> tuple[np.ndarray, list[int]]:
    array = _as_float_array(value, name="Y")
    original_inner_shape = list(array.shape[1:]) if array.ndim > 1 else [1]

    if array.ndim == 1:
        return array.reshape(-1, 1, 1), [1]
    if array.ndim >= 3:
        canonical = array.reshape(array.shape[0], array.shape[1], _safe_prod(tuple(array.shape[2:])))
        return canonical, original_inner_shape

    if expected_horizon and expected_output_dim and array.shape[1] == expected_horizon * expected_output_dim:
        return array.reshape(array.shape[0], expected_horizon, expected_output_dim), original_inner_shape
    if expected_horizon == 1:
        return array.reshape(array.shape[0], 1, array.shape[1]), original_inner_shape
    if expected_output_dim == 1:
        return array.reshape(array.shape[0], array.shape[1], 1), original_inner_shape

    return array.reshape(array.shape[0], 1, array.shape[1]), original_inner_shape


def _flatten_batch(array: np.ndarray) -> np.ndarray:
    return array.reshape(array.shape[0], -1)


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.square(diff)))


class LaggedLinearMethod(BaseMethod):
    """Shared lagged linear regression core used by ARX / VAR wrappers."""

    SUPPORTS_KERNEL_RECOVERY: ClassVar[bool] = False
    MODEL_FAMILY: ClassVar[str] = "lagged_linear"
    FEATURE_ROLE: ClassVar[str] = "lagged_features"

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        alpha: Optional[float] = None,
        fit_intercept: Optional[bool] = None,
    ) -> None:
        merged_config = dict(config or {})
        if alpha is not None:
            merged_config["alpha"] = alpha
        if fit_intercept is not None:
            merged_config["fit_intercept"] = fit_intercept
        super().__init__(config=merged_config, device=device, dtype=dtype)

        self.alpha: float = float(self.config.get("alpha", 1e-6))
        self.fit_intercept: bool = bool(self.config.get("fit_intercept", True))

        self.window_length_: Optional[int] = None
        self.input_dim_: Optional[int] = None
        self.horizon_: Optional[int] = None
        self.output_dim_: Optional[int] = None
        self.feature_dim_: Optional[int] = None
        self.target_dim_: Optional[int] = None
        self.target_inner_shape_: Optional[list[int]] = None

        self.coefficient_matrix_: Optional[np.ndarray] = None
        self.intercept_vector_: Optional[np.ndarray] = None
        self.coefficient_tensor_: Optional[np.ndarray] = None
        self.intercept_tensor_: Optional[np.ndarray] = None

    def _build_estimator(self) -> LinearRegression | Ridge:
        if self.alpha <= 0.0:
            return LinearRegression(fit_intercept=self.fit_intercept)
        return Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)

    def _assert_fitted(self) -> None:
        if not self.is_fitted or self.coefficient_matrix_ is None or self.intercept_vector_ is None:
            raise RuntimeError(f"{self.method_name} must be fitted before use.")

    def _set_dimensions(
        self,
        *,
        window_length: int,
        input_dim: int,
        horizon: int,
        output_dim: int,
        target_inner_shape: list[int],
    ) -> None:
        self.window_length_ = int(window_length)
        self.input_dim_ = int(input_dim)
        self.horizon_ = int(horizon)
        self.output_dim_ = int(output_dim)
        self.feature_dim_ = int(window_length * input_dim)
        self.target_dim_ = int(horizon * output_dim)
        self.target_inner_shape_ = list(target_inner_shape)

    def _refresh_parameter_views(self) -> None:
        if (
            self.coefficient_matrix_ is None
            or self.intercept_vector_ is None
            or self.horizon_ is None
            or self.output_dim_ is None
            or self.window_length_ is None
            or self.input_dim_ is None
        ):
            self.coefficient_tensor_ = None
            self.intercept_tensor_ = None
            return

        self.coefficient_tensor_ = self.coefficient_matrix_.reshape(
            self.horizon_,
            self.output_dim_,
            self.window_length_,
            self.input_dim_,
        )
        self.intercept_tensor_ = self.intercept_vector_.reshape(self.horizon_, self.output_dim_)

    def _predict_flat(self, x_flat: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return np.asarray(x_flat, dtype=np.float64) @ self.coefficient_matrix_.T + self.intercept_vector_

    def _predict_split_array(self, split_x: Any) -> np.ndarray:
        x_canonical = _coerce_feature_batch(
            split_x,
            expected_window_length=self.window_length_,
            expected_input_dim=self.input_dim_,
            allow_single_sample=True,
        )
        x_flat = _flatten_batch(x_canonical)
        if self.feature_dim_ is None or x_flat.shape[1] != self.feature_dim_:
            raise ValueError(
                f"Expected flattened feature width {self.feature_dim_}, got {x_flat.shape[1]}."
            )
        pred_flat = self._predict_flat(x_flat)
        target_inner_shape = tuple(self.target_inner_shape_ or [self.horizon_ or 1, self.output_dim_ or 1])
        return pred_flat.reshape((pred_flat.shape[0],) + target_inner_shape)

    def fit(self, dataset_bundle: Any, **kwargs: Any) -> MethodResult:
        del kwargs
        bundle = self.normalize_dataset_bundle(dataset_bundle)

        train_x = _coerce_feature_batch(
            bundle.train.X,
            expected_window_length=bundle.meta.window_length,
            expected_input_dim=bundle.meta.input_dim,
        )
        train_y, target_inner_shape = _coerce_target_batch(
            bundle.train.Y,
            expected_horizon=bundle.meta.horizon,
            expected_output_dim=bundle.meta.output_dim,
        )
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train X and Y must contain the same number of samples.")

        self._set_dimensions(
            window_length=int(train_x.shape[1]),
            input_dim=int(train_x.shape[2]),
            horizon=int(train_y.shape[1]),
            output_dim=int(train_y.shape[2]),
            target_inner_shape=target_inner_shape,
        )

        train_x_flat = _flatten_batch(train_x)
        train_y_flat = _flatten_batch(train_y)

        estimator = self._build_estimator()
        estimator.fit(train_x_flat, train_y_flat)

        coef = _ensure_2d_coef(np.asarray(estimator.coef_, dtype=np.float64))
        intercept = np.asarray(estimator.intercept_, dtype=np.float64).reshape(-1)
        if intercept.size == 1 and coef.shape[0] > 1:
            intercept = np.full((coef.shape[0],), float(intercept.item()), dtype=np.float64)

        self.coefficient_matrix_ = coef
        self.intercept_vector_ = intercept
        self._refresh_parameter_views()
        self.is_fitted = True

        split_metrics: dict[str, float] = {}
        split_sizes: dict[str, int] = {}
        for split_name in ("train", "val", "test"):
            split = bundle.get_split(split_name)
            split_x = _coerce_feature_batch(
                split.X,
                expected_window_length=self.window_length_,
                expected_input_dim=self.input_dim_,
            )
            split_y, _ = _coerce_target_batch(
                split.Y,
                expected_horizon=self.horizon_,
                expected_output_dim=self.output_dim_,
            )
            split_x_flat = _flatten_batch(split_x)
            split_y_flat = _flatten_batch(split_y)
            split_pred_flat = self._predict_flat(split_x_flat)
            split_metrics[f"{split_name}_mse"] = _mse(split_pred_flat, split_y_flat)
            split_sizes[f"{split_name}_samples"] = int(split_x_flat.shape[0])

        estimator_name = "linear_regression" if self.alpha <= 0.0 else "ridge"
        self.training_summary = {
            "dataset_name": bundle.meta.dataset_name,
            "task_family": bundle.meta.task_family.value,
            "model_family": self.MODEL_FAMILY,
            "feature_role": self.FEATURE_ROLE,
            "estimator": estimator_name,
            "alpha": float(self.alpha),
            "fit_intercept": bool(self.fit_intercept),
            "window_length": int(self.window_length_),
            "input_dim": int(self.input_dim_),
            "horizon": int(self.horizon_),
            "output_dim": int(self.output_dim_),
            "feature_dim": int(self.feature_dim_),
            "target_dim": int(self.target_dim_),
            "coefficient_matrix_shape": list(self.coefficient_matrix_.shape),
            "coefficient_tensor_shape": list(self.coefficient_tensor_.shape) if self.coefficient_tensor_ is not None else None,
            "kernel_recovery_semantics": _RECOVERY_SEMANTICS,
            **split_sizes,
            **split_metrics,
        }

        return MethodResult(
            predictions=None,
            model_state_path=self.model_state_path,
            training_summary=dict(self.training_summary),
            artifacts={},
            metadata={
                "coefficient_matrix_shape": list(self.coefficient_matrix_.shape),
                "intercept_vector_shape": list(self.intercept_vector_.shape),
                "coefficient_tensor_shape": list(self.coefficient_tensor_.shape) if self.coefficient_tensor_ is not None else None,
                "recovery_semantics": _RECOVERY_SEMANTICS,
            },
        )

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        del kwargs
        return self._predict_split_array(X)

    def export_artifacts(self, output_dir: PathLike) -> Mapping[str, Any]:
        self._assert_fitted()

        export_dir = Path(output_dir).expanduser()
        export_dir.mkdir(parents=True, exist_ok=True)

        coefficient_matrix_path = export_dir / "coefficient_matrix.npy"
        intercept_vector_path = export_dir / "intercept_vector.npy"
        coefficient_tensor_path = export_dir / "coefficient_tensor.npy"
        intercept_tensor_path = export_dir / "intercept_tensor.npy"
        metadata_path = export_dir / "linear_model_metadata.json"

        np.save(coefficient_matrix_path, self.coefficient_matrix_)
        np.save(intercept_vector_path, self.intercept_vector_)
        if self.coefficient_tensor_ is not None:
            np.save(coefficient_tensor_path, self.coefficient_tensor_)
        if self.intercept_tensor_ is not None:
            np.save(intercept_tensor_path, self.intercept_tensor_)

        metadata = {
            "method_name": self.method_name,
            "model_family": self.MODEL_FAMILY,
            "feature_role": self.FEATURE_ROLE,
            "equation": "y_flat = x_flat @ coefficient_matrix.T + intercept_vector",
            "coefficient_matrix_shape": list(self.coefficient_matrix_.shape),
            "intercept_vector_shape": list(self.intercept_vector_.shape),
            "coefficient_tensor_shape": list(self.coefficient_tensor_.shape) if self.coefficient_tensor_ is not None else None,
            "intercept_tensor_shape": list(self.intercept_tensor_.shape) if self.intercept_tensor_ is not None else None,
            "window_length": self.window_length_,
            "input_dim": self.input_dim_,
            "horizon": self.horizon_,
            "output_dim": self.output_dim_,
            "target_inner_shape": list(self.target_inner_shape_ or []),
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "kernel_recovery_semantics": _RECOVERY_SEMANTICS,
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        return {
            "coefficient_matrix": ArtifactRef(
                kind="npy",
                path=str(coefficient_matrix_path),
                metadata={
                    "shape": list(self.coefficient_matrix_.shape),
                    "equation": metadata["equation"],
                },
            ),
            "intercept_vector": ArtifactRef(
                kind="npy",
                path=str(intercept_vector_path),
                metadata={"shape": list(self.intercept_vector_.shape)},
            ),
            "coefficient_tensor": ArtifactRef(
                kind="npy",
                path=str(coefficient_tensor_path),
                metadata={
                    "shape": list(self.coefficient_tensor_.shape) if self.coefficient_tensor_ is not None else None,
                    "axes": ["horizon", "output_channel", "lag", "input_channel"],
                },
            ),
            "intercept_tensor": ArtifactRef(
                kind="npy",
                path=str(intercept_tensor_path),
                metadata={
                    "shape": list(self.intercept_tensor_.shape) if self.intercept_tensor_ is not None else None,
                    "axes": ["horizon", "output_channel"],
                },
            ),
            "metadata": ArtifactRef(
                kind="json",
                path=str(metadata_path),
                metadata={"model_family": self.MODEL_FAMILY},
            ),
        }

    def get_state(self) -> Mapping[str, Any]:
        return {
            "alpha": float(self.alpha),
            "fit_intercept": bool(self.fit_intercept),
            "window_length": self.window_length_,
            "input_dim": self.input_dim_,
            "horizon": self.horizon_,
            "output_dim": self.output_dim_,
            "feature_dim": self.feature_dim_,
            "target_dim": self.target_dim_,
            "target_inner_shape": list(self.target_inner_shape_ or []),
            "coefficient_matrix": self.coefficient_matrix_.tolist() if self.coefficient_matrix_ is not None else None,
            "intercept_vector": self.intercept_vector_.tolist() if self.intercept_vector_ is not None else None,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        self.alpha = float(state.get("alpha", self.config.get("alpha", 1e-6)))
        self.fit_intercept = bool(state.get("fit_intercept", self.config.get("fit_intercept", True)))

        self.window_length_ = int(state["window_length"]) if state.get("window_length") is not None else None
        self.input_dim_ = int(state["input_dim"]) if state.get("input_dim") is not None else None
        self.horizon_ = int(state["horizon"]) if state.get("horizon") is not None else None
        self.output_dim_ = int(state["output_dim"]) if state.get("output_dim") is not None else None
        self.feature_dim_ = int(state["feature_dim"]) if state.get("feature_dim") is not None else None
        self.target_dim_ = int(state["target_dim"]) if state.get("target_dim") is not None else None

        target_inner_shape = state.get("target_inner_shape", [])
        self.target_inner_shape_ = [int(v) for v in target_inner_shape] if target_inner_shape else None

        coef_payload = state.get("coefficient_matrix")
        intercept_payload = state.get("intercept_vector")
        self.coefficient_matrix_ = (
            np.asarray(coef_payload, dtype=np.float64) if coef_payload is not None else None
        )
        self.intercept_vector_ = (
            np.asarray(intercept_payload, dtype=np.float64).reshape(-1) if intercept_payload is not None else None
        )
        if self.coefficient_matrix_ is not None:
            self.coefficient_matrix_ = _ensure_2d_coef(self.coefficient_matrix_)
        self._refresh_parameter_views()


@register_method("arx", overwrite=True)
class ARXMethod(LaggedLinearMethod):
    """ARX-style lagged linear baseline using the provided X window as exogenous history."""

    MODEL_FAMILY: ClassVar[str] = "arx"
    FEATURE_ROLE: ClassVar[str] = "exogenous_history"


@register_method("var", overwrite=True)
class VARMethod(LaggedLinearMethod):
    """VAR-style lagged linear baseline using the provided X window as state history."""

    MODEL_FAMILY: ClassVar[str] = "var"
    FEATURE_ROLE: ClassVar[str] = "state_history"
