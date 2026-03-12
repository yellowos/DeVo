"""Laguerre-Volterra baseline compatible with the unified methods interface."""

from __future__ import annotations

import json
from itertools import combinations_with_replacement, permutations
from math import comb, factorial, sqrt
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from methods.base import ArtifactRef, BaseMethod, KernelRecoveryResult, MethodResult, register_method


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _impulse_response(numerator: np.ndarray, denominator: np.ndarray, length: int) -> np.ndarray:
    response = np.zeros(length, dtype=np.float64)
    for index in range(length):
        value = numerator[index] if index < len(numerator) else 0.0
        for delay in range(1, min(index + 1, len(denominator))):
            value -= denominator[delay] * response[index - delay]
        response[index] = value / denominator[0]
    return response


def build_discrete_laguerre_basis(
    window_length: int,
    num_functions: int,
    pole: float,
) -> np.ndarray:
    """Generate a truncated discrete Laguerre basis over lag indices 0..L-1."""

    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if num_functions <= 0:
        raise ValueError("num_functions must be positive.")
    if not 0.0 < pole < 1.0:
        raise ValueError("pole must lie in (0, 1).")

    basis = np.zeros((num_functions, window_length), dtype=np.float64)
    scale = sqrt(1.0 - pole**2)
    for order in range(num_functions):
        numerator = np.array(
            [comb(order, degree) * ((-pole) ** (order - degree)) for degree in range(order + 1)],
            dtype=np.float64,
        )
        numerator *= scale
        denominator = np.array(
            [comb(order + 1, degree) * ((-pole) ** degree) for degree in range(order + 2)],
            dtype=np.float64,
        )
        basis[order] = _impulse_response(numerator, denominator, window_length)
    return basis


def _permutation_count(indices: Sequence[int]) -> int:
    multiplicities: dict[int, int] = {}
    for index in indices:
        multiplicities[index] = multiplicities.get(index, 0) + 1
    denominator = 1
    for count in multiplicities.values():
        denominator *= factorial(count)
    return factorial(len(indices)) // denominator


def _flatten_target_shape(target_shape: Sequence[int]) -> int:
    size = 1
    for dim in target_shape:
        size *= int(dim)
    return size


@register_method("laguerre_volterra")
class LaguerreVolterraMethod(BaseMethod):
    """Laguerre basis expansion followed by Volterra polynomial regression."""

    SUPPORTS_KERNEL_RECOVERY = True

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        super().__init__(config=config, device=device, dtype=dtype)
        self.laguerre_pole = float(self.config.get("laguerre_pole", 0.6))
        self.num_laguerre = int(self.config.get("num_laguerre", 6))
        self.volterra_order = int(self.config.get("volterra_order", 2))
        self.ridge_lambda = float(self.config.get("ridge_lambda", 1e-6))
        self.include_bias = bool(self.config.get("include_bias", True))
        self.max_feature_terms = int(self.config.get("max_feature_terms", 4096))

        if not 0.0 < self.laguerre_pole < 1.0:
            raise ValueError("laguerre_pole must lie in (0, 1).")
        if self.num_laguerre <= 0:
            raise ValueError("num_laguerre must be positive.")
        if self.volterra_order <= 0:
            raise ValueError("volterra_order must be positive.")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be non-negative.")
        if self.max_feature_terms <= 0:
            raise ValueError("max_feature_terms must be positive.")

        self.window_length: Optional[int] = None
        self.input_dim: Optional[int] = None
        self.target_shape: Optional[tuple[int, ...]] = None
        self.laguerre_basis_: Optional[np.ndarray] = None
        self.regression_weights_: Optional[np.ndarray] = None
        self.base_index_map_: list[dict[str, int]] = []
        self.feature_specs_: list[tuple[int, ...]] = []

    def _effective_config(self) -> dict[str, Any]:
        return {
            "laguerre_pole": self.laguerre_pole,
            "num_laguerre": self.num_laguerre,
            "volterra_order": self.volterra_order,
            "ridge_lambda": self.ridge_lambda,
            "include_bias": self.include_bias,
            "max_feature_terms": self.max_feature_terms,
        }

    def _validate_fitted(self) -> None:
        if not self.is_fitted or self.regression_weights_ is None or self.laguerre_basis_ is None:
            raise RuntimeError("LaguerreVolterraMethod must be fitted before use.")
        if self.window_length is None or self.input_dim is None or self.target_shape is None:
            raise RuntimeError("Fitted model is missing structural metadata.")

    def _coerce_target_array(self, Y: Any) -> np.ndarray:
        target = np.asarray(Y, dtype=np.float64)
        if target.ndim == 1:
            target = target[:, np.newaxis]
        if target.ndim < 2:
            raise ValueError("Y must have at least a batch dimension.")
        return target

    def _coerce_input_batch(self, X: Any) -> np.ndarray:
        self._validate_fitted()
        assert self.window_length is not None
        assert self.input_dim is not None

        batch = np.asarray(X, dtype=np.float64)
        if batch.ndim == 1:
            if self.input_dim != 1:
                raise ValueError("1D X is only supported for single-input models.")
            batch = batch[np.newaxis, :, np.newaxis]
        elif batch.ndim == 2:
            if batch.shape == (self.window_length, self.input_dim):
                batch = batch[np.newaxis, :, :]
            elif self.input_dim == 1 and batch.shape[1] == self.window_length:
                batch = batch[:, :, np.newaxis]
            else:
                raise ValueError(
                    "2D X must be shaped as (window_length, input_dim) for one sample "
                    "or (num_samples, window_length) for single-input models."
                )
        elif batch.ndim != 3:
            raise ValueError("X must be 1D, 2D, or 3D.")

        if batch.shape[1] != self.window_length:
            raise ValueError(f"Expected window_length={self.window_length}, got {batch.shape[1]}.")
        if batch.shape[2] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {batch.shape[2]}.")
        return batch

    def _initialize_structure_from_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.window_length = int(X.shape[1])
        self.input_dim = int(X.shape[2])
        self.target_shape = tuple(int(dim) for dim in Y.shape[1:]) or (1,)
        self.laguerre_basis_ = build_discrete_laguerre_basis(
            window_length=self.window_length,
            num_functions=self.num_laguerre,
            pole=self.laguerre_pole,
        )
        self.base_index_map_ = [
            {
                "flat_index": channel * self.num_laguerre + basis_index,
                "input_channel": channel,
                "laguerre_index": basis_index,
            }
            for channel in range(self.input_dim)
            for basis_index in range(self.num_laguerre)
        ]
        self.feature_specs_ = []
        if self.include_bias:
            self.feature_specs_.append(tuple())
        num_base_terms = self.input_dim * self.num_laguerre
        for order in range(1, self.volterra_order + 1):
            for spec in combinations_with_replacement(range(num_base_terms), order):
                self.feature_specs_.append(tuple(spec))
        if len(self.feature_specs_) > self.max_feature_terms:
            raise ValueError(
                "Laguerre-Volterra feature count exceeds max_feature_terms: "
                f"{len(self.feature_specs_)} > {self.max_feature_terms}"
            )

    def _laguerre_states(self, X: np.ndarray) -> np.ndarray:
        if self.laguerre_basis_ is None:
            raise RuntimeError("Laguerre basis has not been initialized.")
        lag_ordered = X[:, ::-1, :]
        return np.einsum("nlc,kl->nck", lag_ordered, self.laguerre_basis_, optimize=True)

    def _build_design_matrix(self, X: np.ndarray) -> np.ndarray:
        states = self._laguerre_states(X)
        base_terms = states.reshape(states.shape[0], -1)
        design = np.empty((base_terms.shape[0], len(self.feature_specs_)), dtype=np.float64)
        for column, spec in enumerate(self.feature_specs_):
            if len(spec) == 0:
                design[:, column] = 1.0
                continue
            design[:, column] = np.prod(base_terms[:, spec], axis=1)
        return design

    def _solve_regression(self, design: np.ndarray, targets: np.ndarray) -> np.ndarray:
        if self.ridge_lambda > 0.0:
            diagonal = np.full(design.shape[1], sqrt(self.ridge_lambda), dtype=np.float64)
            if self.include_bias and diagonal.size:
                diagonal[0] = 0.0
            regularizer = np.diag(diagonal)
            augmented_design = np.vstack([design, regularizer])
            augmented_targets = np.vstack([targets, np.zeros((design.shape[1], targets.shape[1]), dtype=np.float64)])
            weights, *_ = np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)
            return weights
        weights, *_ = np.linalg.lstsq(design, targets, rcond=None)
        return weights

    def fit(self, dataset_bundle: Any, **kwargs: Any) -> MethodResult:
        bundle = self.normalize_dataset_bundle(dataset_bundle, dataset_name=kwargs.get("dataset_name"))
        train_X = np.asarray(bundle.train.X, dtype=np.float64)
        train_Y = self._coerce_target_array(bundle.train.Y)

        if train_X.ndim == 2:
            train_X = train_X[:, :, np.newaxis]
        if train_X.ndim != 3:
            raise ValueError("Training X must be shaped as (num_samples, window_length, input_dim).")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("Training X and Y must share the same number of samples.")

        self._initialize_structure_from_data(train_X, train_Y)
        design = self._build_design_matrix(train_X)
        flat_targets = train_Y.reshape(train_Y.shape[0], -1)
        self.regression_weights_ = self._solve_regression(design, flat_targets)
        self.training_summary = {
            "dataset_name": bundle.meta.dataset_name,
            "train_samples": int(train_X.shape[0]),
            "window_length": int(train_X.shape[1]),
            "input_dim": int(train_X.shape[2]),
            "target_shape": list(self.target_shape or ()),
            "num_laguerre": self.num_laguerre,
            "volterra_order": self.volterra_order,
            "feature_count": len(self.feature_specs_),
            "supports_kernel_recovery": True,
        }
        self.is_fitted = True
        return MethodResult(
            predictions=None,
            model_state_path=self.model_state_path,
            training_summary=dict(self.training_summary),
            artifacts={},
            metadata={"representation": "laguerre_volterra"},
        )

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        del kwargs
        batch = self._coerce_input_batch(X)
        design = self._build_design_matrix(batch)
        assert self.regression_weights_ is not None
        flat_predictions = design @ self.regression_weights_
        assert self.target_shape is not None
        return flat_predictions.reshape((batch.shape[0],) + self.target_shape)

    def _build_order_dense_tensor(self, order: int) -> np.ndarray:
        self._validate_fitted()
        assert self.regression_weights_ is not None
        assert self.target_shape is not None
        if order == 0:
            if not self.include_bias or not self.feature_specs_ or len(self.feature_specs_[0]) != 0:
                return np.zeros(self.target_shape, dtype=np.float64)
            return self.regression_weights_[0].reshape(self.target_shape)

        num_base_terms = len(self.base_index_map_)
        flat_target_dim = _flatten_target_shape(self.target_shape)
        dense = np.zeros((flat_target_dim,) + (num_base_terms,) * order, dtype=np.float64)
        for feature_index, spec in enumerate(self.feature_specs_):
            if len(spec) != order:
                continue
            coefficient = self.regression_weights_[feature_index].reshape(flat_target_dim)
            permutation_weight = float(_permutation_count(spec))
            for permuted in set(permutations(spec)):
                dense[(slice(None),) + permuted] += coefficient / permutation_weight
        return dense.reshape(self.target_shape + (self.input_dim, self.num_laguerre) * order)

    def _identified_terms_for_order(self, order: int) -> list[dict[str, Any]]:
        self._validate_fitted()
        assert self.regression_weights_ is not None
        assert self.target_shape is not None
        terms: list[dict[str, Any]] = []
        for feature_index, spec in enumerate(self.feature_specs_):
            if len(spec) != order:
                continue
            factors = [self.base_index_map_[base_index] for base_index in spec]
            terms.append(
                {
                    "feature_index": feature_index,
                    "basis_factors": factors,
                    "coefficient": self.regression_weights_[feature_index].reshape(self.target_shape).copy(),
                }
            )
        return terms

    def _time_domain_kernel_map(self) -> dict[str, np.ndarray]:
        self._validate_fitted()
        assert self.laguerre_basis_ is not None

        mapped: dict[str, np.ndarray] = {
            "order_0": self._build_order_dense_tensor(0),
        }
        if self.volterra_order >= 1:
            order_1 = self._build_order_dense_tensor(1)
            mapped["order_1"] = np.einsum("...ck,kl->...cl", order_1, self.laguerre_basis_, optimize=True)
        if self.volterra_order >= 2:
            order_2 = self._build_order_dense_tensor(2)
            mapped["order_2"] = np.einsum(
                "...aibj,im,jn->...ambn",
                order_2,
                self.laguerre_basis_,
                self.laguerre_basis_,
                optimize=True,
            )
        return mapped

    def recover_kernels(
        self,
        *,
        map_to_time_domain: bool = True,
        include_identified_terms: bool = True,
        **kwargs: Any,
    ) -> KernelRecoveryResult:
        del kwargs
        self._validate_fitted()
        assert self.target_shape is not None

        orders: dict[str, Any] = {}
        for order in range(0, self.volterra_order + 1):
            entry: dict[str, Any] = {
                "coefficients": self._build_order_dense_tensor(order),
                "semantics": (
                    "Dense symmetric coefficient tensor in the discrete Laguerre basis. "
                    "Each order-r tensor is defined so that summing it over all Laguerre state "
                    "indices reproduces the fitted polynomial output."
                ),
            }
            if include_identified_terms:
                entry["identified_terms"] = self._identified_terms_for_order(order)
            orders[str(order)] = entry

        kernels: dict[str, Any] = {
            "representation": {
                "type": "laguerre_volterra_basis_coefficients",
                "basis_family": "discrete_laguerre",
                "laguerre_pole": self.laguerre_pole,
                "num_laguerre": self.num_laguerre,
                "window_length": self.window_length,
                "input_dim": self.input_dim,
                "target_shape": list(self.target_shape),
                "volterra_order": self.volterra_order,
                "lag_index_semantics": "lag_index=0 corresponds to the most recent sample in the input window",
            },
            "base_variables": list(self.base_index_map_),
            "orders": orders,
        }
        if map_to_time_domain:
            kernels["time_domain_kernels"] = self._time_domain_kernel_map()

        return KernelRecoveryResult(
            kernels=kernels,
            summary={
                "supports_laguerre_basis_coefficients": True,
                "supports_time_domain_mapping": map_to_time_domain and self.volterra_order >= 1,
                "time_domain_mapping_orders": [order for order in (0, 1, 2) if order <= self.volterra_order],
            },
            artifacts={},
        )

    def export_artifacts(self, output_dir: str | Path) -> Mapping[str, Any]:
        self._validate_fitted()
        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        recovery = self.recover_kernels(map_to_time_domain=True, include_identified_terms=True)
        arrays_path = target_dir / "laguerre_volterra_coefficients.npz"
        basis_path = target_dir / "laguerre_basis.npy"
        summary_path = target_dir / "laguerre_volterra_summary.json"
        structure_path = target_dir / "laguerre_feature_structure.json"

        assert self.laguerre_basis_ is not None
        np.save(basis_path, self.laguerre_basis_)

        orders = recovery.kernels["orders"]
        arrays: dict[str, np.ndarray] = {
            "regression_weights": self.regression_weights_,
            "laguerre_basis": self.laguerre_basis_,
        }
        for order, payload in orders.items():
            coefficient_array = payload.get("coefficients")
            if isinstance(coefficient_array, np.ndarray):
                arrays[f"order_{order}_laguerre_coefficients"] = coefficient_array
        for order, kernel in recovery.kernels.get("time_domain_kernels", {}).items():
            if isinstance(kernel, np.ndarray):
                arrays[f"{order}_time_domain_kernel"] = kernel
        np.savez(arrays_path, **arrays)

        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "method_name": self.method_name,
                    "config": self._effective_config(),
                    "training_summary": self.training_summary,
                    "representation": recovery.kernels["representation"],
                    "exported_files": {
                        "basis": str(basis_path),
                        "arrays": str(arrays_path),
                        "feature_structure": str(structure_path),
                    },
                },
                handle,
                indent=2,
            )

        with structure_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "base_variables": self.base_index_map_,
                    "orders": {
                        order: {
                            "semantics": payload.get("semantics"),
                            "identified_terms": _json_safe(payload.get("identified_terms", [])),
                        }
                        for order, payload in orders.items()
                    },
                },
                handle,
                indent=2,
            )

        return {
            "summary": ArtifactRef(kind="json", path=str(summary_path)),
            "basis": ArtifactRef(kind="npy", path=str(basis_path)),
            "coefficients": ArtifactRef(kind="npz", path=str(arrays_path)),
            "feature_structure": ArtifactRef(kind="json", path=str(structure_path)),
        }

    def get_state(self) -> Mapping[str, Any]:
        return {
            "laguerre_pole": self.laguerre_pole,
            "num_laguerre": self.num_laguerre,
            "volterra_order": self.volterra_order,
            "ridge_lambda": self.ridge_lambda,
            "include_bias": self.include_bias,
            "max_feature_terms": self.max_feature_terms,
            "window_length": self.window_length,
            "input_dim": self.input_dim,
            "target_shape": list(self.target_shape) if self.target_shape is not None else None,
            "laguerre_basis": _json_safe(self.laguerre_basis_),
            "regression_weights": _json_safe(self.regression_weights_),
            "base_index_map": list(self.base_index_map_),
            "feature_specs": [list(spec) for spec in self.feature_specs_],
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        self.laguerre_pole = float(state.get("laguerre_pole", self.laguerre_pole))
        self.num_laguerre = int(state.get("num_laguerre", self.num_laguerre))
        self.volterra_order = int(state.get("volterra_order", self.volterra_order))
        self.ridge_lambda = float(state.get("ridge_lambda", self.ridge_lambda))
        self.include_bias = bool(state.get("include_bias", self.include_bias))
        self.max_feature_terms = int(state.get("max_feature_terms", self.max_feature_terms))
        self.window_length = (
            None if state.get("window_length") is None else int(state.get("window_length"))
        )
        self.input_dim = None if state.get("input_dim") is None else int(state.get("input_dim"))
        target_shape = state.get("target_shape")
        self.target_shape = None if target_shape is None else tuple(int(dim) for dim in target_shape)

        laguerre_basis = state.get("laguerre_basis")
        self.laguerre_basis_ = None if laguerre_basis is None else np.asarray(laguerre_basis, dtype=np.float64)
        regression_weights = state.get("regression_weights")
        self.regression_weights_ = (
            None if regression_weights is None else np.asarray(regression_weights, dtype=np.float64)
        )
        self.base_index_map_ = [
            {
                "flat_index": int(entry["flat_index"]),
                "input_channel": int(entry["input_channel"]),
                "laguerre_index": int(entry["laguerre_index"]),
            }
            for entry in state.get("base_index_map", [])
        ]
        self.feature_specs_ = [tuple(int(index) for index in spec) for spec in state.get("feature_specs", [])]
