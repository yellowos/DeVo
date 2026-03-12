"""CP-Volterra baseline with a unified methods-layer interface.

The model uses a truncated Volterra series where each order-k kernel is
parameterized by a CP decomposition. Prediction never materializes the full
kernel tensor; instead it evaluates CP projections batch by batch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from methods.base import ArtifactRef, BaseMethod, KernelRecoveryResult, MethodResult, register_method
from methods.base.io_schema import DatasetBundleSource

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is absent.
    torch = None

if torch is None:  # pragma: no cover - exercised only when torch is absent.
    _TorchModuleBase = object
else:
    _TorchModuleBase = torch.nn.Module


PathLike = str | Path
_DEFAULT_KERNEL_EXPANSION_LIMIT = 1_000_000
_EPS = 1e-8


def _ensure_torch() -> None:
    if torch is None:
        raise RuntimeError("CPVolterraMethod requires PyTorch to be installed.")


def _to_numpy_float(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    return array


def _safe_scale(array: np.ndarray) -> np.ndarray:
    scale = np.std(array, axis=0, ddof=0)
    scale = np.asarray(scale, dtype=np.float32)
    scale[~np.isfinite(scale)] = 1.0
    scale[np.abs(scale) < _EPS] = 1.0
    return scale


def _mse_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(y_true - y_pred)))


def _iter_batches(num_samples: int, batch_size: int, shuffle: bool, rng: np.random.Generator) -> Iterable[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    indices = np.arange(num_samples)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        yield indices[start : start + batch_size]


def _flatten_output_shape(output_shape: Sequence[int]) -> int:
    size = 1
    for dim in output_shape:
        size *= int(dim)
    return int(size)


def _as_tuple_of_ints(value: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


class _CPVolterraModule(_TorchModuleBase):
    """Torch module for CP-parameterized Volterra regression."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        order_ranks: Mapping[int, int],
        init_scale: float,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.orders = tuple(sorted(int(order) for order in order_ranks))
        self.bias = torch.nn.Parameter(torch.zeros(self.output_size))
        self.output_weights = torch.nn.ParameterDict()
        self.input_factors = torch.nn.ParameterDict()

        factor_std = float(init_scale) / max(math.sqrt(self.input_size), 1.0)
        for order in self.orders:
            rank = int(order_ranks[order])
            if rank <= 0:
                raise ValueError(f"Rank for order {order} must be positive.")
            output_key = self._output_key(order)
            self.output_weights[output_key] = torch.nn.Parameter(
                torch.randn(self.output_size, rank) * factor_std
            )
            for mode in range(order):
                factor_key = self._factor_key(order, mode)
                self.input_factors[factor_key] = torch.nn.Parameter(
                    torch.randn(rank, self.input_size) * factor_std
                )

    @staticmethod
    def _output_key(order: int) -> str:
        return f"order_{order}_output"

    @staticmethod
    def _factor_key(order: int, mode: int) -> str:
        return f"order_{order}_factor_{mode}"

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        output = self.bias.expand(x_flat.shape[0], -1)
        for order in self.orders:
            output_weights = self.output_weights[self._output_key(order)]
            term: Optional[torch.Tensor] = None
            for mode in range(order):
                factor = self.input_factors[self._factor_key(order, mode)]
                projection = x_flat @ factor.transpose(0, 1)
                term = projection if term is None else term * projection
            if term is None:
                continue
            output = output + term @ output_weights.transpose(0, 1)
        return output


@register_method("cp_volterra")
class CPVolterraMethod(BaseMethod):
    """Low-rank CP Volterra baseline for nonlinear benchmark regression."""

    SUPPORTS_KERNEL_RECOVERY = True

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        super().__init__(config=config, device=device, dtype=dtype)
        self.max_order: int = 0
        self.order_ranks: dict[int, int] = {}
        self.window_length: Optional[int] = None
        self.input_dim: Optional[int] = None
        self.output_shape: tuple[int, ...] = ()
        self.output_size: int = 0
        self.feature_size: int = 0
        self.x_scale: Optional[np.ndarray] = None
        self.y_scale: Optional[np.ndarray] = None
        self.y_mean: Optional[np.ndarray] = None
        self.dataset_metadata: dict[str, Any] = {}
        self._model: Optional[_CPVolterraModule] = None
        self._last_kernel_recovery: Optional[KernelRecoveryResult] = None

    def _resolve_order_ranks(self, max_order: int, *, order_ranks: Any = None) -> dict[int, int]:
        if max_order <= 0:
            raise ValueError("max_order must be positive.")
        raw = order_ranks
        if raw is None:
            raw = self.config.get("order_ranks", self.config.get("rank", 4))

        if isinstance(raw, Mapping):
            ranks = {order: int(raw.get(order, raw.get(str(order), 0))) for order in range(1, max_order + 1)}
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = [int(item) for item in raw]
            if len(values) < max_order:
                raise ValueError("order_ranks sequence length must cover max_order.")
            ranks = {order: values[order - 1] for order in range(1, max_order + 1)}
        else:
            rank_value = int(raw)
            ranks = {order: rank_value for order in range(1, max_order + 1)}

        filtered = {order: rank for order, rank in ranks.items() if rank > 0}
        if not filtered:
            raise ValueError("At least one positive CP rank is required.")
        return filtered

    def _coerce_input_array(
        self,
        X: Any,
        *,
        expected_window_length: Optional[int] = None,
        expected_input_dim: Optional[int] = None,
    ) -> np.ndarray:
        array = _to_numpy_float(X, name="X")
        if array.ndim == 1:
            if expected_window_length and expected_input_dim:
                expected_size = int(expected_window_length) * int(expected_input_dim)
                if array.size != expected_size:
                    raise ValueError(
                        f"1D X length {array.size} does not match expected feature size {expected_size}."
                    )
                return array.reshape(1, int(expected_window_length), int(expected_input_dim))
            return array.reshape(1, array.shape[0], 1)

        if array.ndim == 2:
            if expected_window_length and expected_input_dim:
                expected_size = int(expected_window_length) * int(expected_input_dim)
                if array.shape == (int(expected_window_length), int(expected_input_dim)):
                    return array.reshape(1, int(expected_window_length), int(expected_input_dim))
                if array.shape[1] == expected_size:
                    return array.reshape(array.shape[0], int(expected_window_length), int(expected_input_dim))
                if int(expected_input_dim) == 1 and array.shape[1] == int(expected_window_length):
                    return array.reshape(array.shape[0], int(expected_window_length), 1)
                raise ValueError(
                    "2D X does not match fitted window_length/input_dim and cannot be reshaped safely."
                )
            return array.reshape(array.shape[0], array.shape[1], 1)

        if array.ndim == 3:
            if expected_window_length and array.shape[1] != int(expected_window_length):
                raise ValueError(
                    f"X window_length mismatch: expected {expected_window_length}, got {array.shape[1]}."
                )
            if expected_input_dim and array.shape[2] != int(expected_input_dim):
                raise ValueError(
                    f"X input_dim mismatch: expected {expected_input_dim}, got {array.shape[2]}."
                )
            return array

        raise ValueError("X must be a 1D, 2D, or 3D numeric array.")

    def _coerce_target_array(
        self,
        Y: Any,
        *,
        expected_horizon: Optional[int] = None,
        expected_output_dim: Optional[int] = None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        raw = _to_numpy_float(Y, name="Y")
        if raw.ndim == 1:
            flat = raw.reshape(raw.shape[0], 1)
        else:
            flat = raw.reshape(raw.shape[0], -1)

        if expected_horizon and expected_output_dim:
            shape = (int(expected_horizon), int(expected_output_dim))
            expected_size = _flatten_output_shape(shape)
            if flat.shape[1] != expected_size:
                raise ValueError(
                    f"Y feature size mismatch: expected {expected_size}, got {flat.shape[1]}."
                )
            return flat, shape

        if raw.ndim == 1:
            return flat, (1,)
        return flat, _as_tuple_of_ints(raw.shape[1:])

    def _require_fitted(self) -> None:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("CPVolterraMethod must be fitted before use.")

    def _require_model(self) -> None:
        if self._model is None:
            raise RuntimeError("CPVolterraMethod has no initialized torch model.")

    def _build_model(self) -> _CPVolterraModule:
        _ensure_torch()
        if self.feature_size <= 0 or self.output_size <= 0 or not self.order_ranks:
            raise RuntimeError("Model dimensions are incomplete.")
        init_scale = float(self.config.get("init_scale", 0.05))
        model = _CPVolterraModule(
            input_size=self.feature_size,
            output_size=self.output_size,
            order_ranks=self.order_ranks,
            init_scale=init_scale,
        )
        return model.to(device=self.runtime.device, dtype=self.runtime.dtype)

    def _serialize_weight_arrays(self) -> Dict[str, np.ndarray]:
        self._require_model()
        assert self._model is not None
        payload: Dict[str, np.ndarray] = {
            "bias": self._model.bias.detach().cpu().numpy(),
            "x_scale": np.asarray(self.x_scale, dtype=np.float32),
            "y_scale": np.asarray(self.y_scale, dtype=np.float32),
            "y_mean": np.asarray(self.y_mean, dtype=np.float32),
        }
        for order in sorted(self.order_ranks):
            output_key = _CPVolterraModule._output_key(order)
            payload[f"order_{order}_output_weights"] = (
                self._model.output_weights[output_key].detach().cpu().numpy()
            )
            for mode in range(order):
                factor_key = _CPVolterraModule._factor_key(order, mode)
                payload[f"order_{order}_factor_{mode}"] = (
                    self._model.input_factors[factor_key].detach().cpu().numpy()
                )
        return payload

    def _load_weight_arrays(self, weights_path: Path) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing CP-Volterra weight file: {weights_path}")
        self._model = self._build_model()
        assert self._model is not None
        with np.load(weights_path, allow_pickle=False) as arrays:
            self.x_scale = np.asarray(arrays["x_scale"], dtype=np.float32)
            self.y_scale = np.asarray(arrays["y_scale"], dtype=np.float32)
            self.y_mean = np.asarray(arrays["y_mean"], dtype=np.float32)
            with torch.no_grad():
                self._model.bias.copy_(
                    torch.as_tensor(arrays["bias"], device=self.runtime.device, dtype=self.runtime.dtype)
                )
                for order in sorted(self.order_ranks):
                    output_key = _CPVolterraModule._output_key(order)
                    self._model.output_weights[output_key].copy_(
                        torch.as_tensor(
                            arrays[f"order_{order}_output_weights"],
                            device=self.runtime.device,
                            dtype=self.runtime.dtype,
                        )
                    )
                    for mode in range(order):
                        factor_key = _CPVolterraModule._factor_key(order, mode)
                        self._model.input_factors[factor_key].copy_(
                            torch.as_tensor(
                                arrays[f"order_{order}_factor_{mode}"],
                                device=self.runtime.device,
                                dtype=self.runtime.dtype,
                            )
                        )

    def _prepare_features(self, X: Any) -> np.ndarray:
        array = self._coerce_input_array(
            X,
            expected_window_length=self.window_length,
            expected_input_dim=self.input_dim,
        )
        flat = array.reshape(array.shape[0], -1)
        if flat.shape[1] != self.feature_size:
            raise ValueError(f"Expected flattened feature size {self.feature_size}, got {flat.shape[1]}.")
        assert self.x_scale is not None
        return (flat / self.x_scale).astype(np.float32, copy=False)

    def _predict_flat_scaled(self, x_scaled: np.ndarray, *, batch_size: int) -> np.ndarray:
        self._require_model()
        assert self._model is not None
        outputs: list[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for idx in _iter_batches(
                num_samples=x_scaled.shape[0],
                batch_size=batch_size,
                shuffle=False,
                rng=np.random.default_rng(0),
            ):
                xb = torch.as_tensor(x_scaled[idx], device=self.runtime.device, dtype=self.runtime.dtype)
                pred = self._model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
                outputs.append(pred)
        if not outputs:
            return np.zeros((0, self.output_size), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _predict_flat_raw(self, x_scaled: np.ndarray, *, batch_size: int) -> np.ndarray:
        scaled = self._predict_flat_scaled(x_scaled, batch_size=batch_size)
        assert self.y_scale is not None and self.y_mean is not None
        return scaled * self.y_scale.reshape(1, -1) + self.y_mean.reshape(1, -1)

    def _predict_split(self, X: Any, *, batch_size: Optional[int] = None) -> np.ndarray:
        self._require_fitted()
        batch = int(batch_size or self.config.get("eval_batch_size", self.config.get("batch_size", 512)))
        features = self._prepare_features(X)
        flat = self._predict_flat_raw(features, batch_size=batch)
        return flat.reshape((flat.shape[0],) + self.output_shape)

    def _evaluate_loss(self, x_scaled: np.ndarray, y_scaled: np.ndarray, *, batch_size: int) -> float:
        pred = self._predict_flat_scaled(x_scaled, batch_size=batch_size)
        return _mse_numpy(y_scaled, pred)

    def fit(self, dataset_bundle: DatasetBundleSource, **kwargs: Any) -> MethodResult:
        _ensure_torch()
        dataset_name = kwargs.get("dataset_name")
        bundle = self.normalize_dataset_bundle(dataset_bundle, dataset_name=dataset_name)

        train_x = self._coerce_input_array(
            bundle.train.X,
            expected_window_length=bundle.meta.window_length,
            expected_input_dim=bundle.meta.input_dim,
        )
        train_y, output_shape = self._coerce_target_array(
            bundle.train.Y,
            expected_horizon=bundle.meta.horizon,
            expected_output_dim=bundle.meta.output_dim,
        )
        val_x = self._coerce_input_array(
            bundle.val.X,
            expected_window_length=bundle.meta.window_length,
            expected_input_dim=bundle.meta.input_dim,
        )
        val_y, _ = self._coerce_target_array(
            bundle.val.Y,
            expected_horizon=bundle.meta.horizon,
            expected_output_dim=bundle.meta.output_dim,
        )
        if train_x.shape[0] == 0:
            raise ValueError("Training split is empty.")
        if val_x.shape[0] == 0:
            raise ValueError(
                "Validation split is empty.  CP-Volterra requires a non-empty "
                "validation set for early-stopping model selection; falling back "
                "to training data would leak information and bias results."
            )

        self.window_length = int(train_x.shape[1])
        self.input_dim = int(train_x.shape[2])
        self.feature_size = int(self.window_length * self.input_dim)
        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.output_size = _flatten_output_shape(self.output_shape)
        max_order = int(kwargs.get("max_order", self.config.get("max_order", 2)))
        self.order_ranks = self._resolve_order_ranks(
            max_order,
            order_ranks=kwargs.get("order_ranks", self.config.get("order_ranks")),
        )
        self.max_order = int(max(self.order_ranks))

        train_x_flat = train_x.reshape(train_x.shape[0], -1)
        val_x_flat = val_x.reshape(val_x.shape[0], -1)

        self.x_scale = _safe_scale(train_x_flat)
        self.y_scale = _safe_scale(train_y)
        self.y_mean = np.mean(train_y, axis=0, dtype=np.float32)

        train_x_scaled = (train_x_flat / self.x_scale).astype(np.float32, copy=False)
        val_x_scaled = (val_x_flat / self.x_scale).astype(np.float32, copy=False)
        train_y_scaled = ((train_y - self.y_mean) / self.y_scale).astype(np.float32, copy=False)
        val_y_scaled = ((val_y - self.y_mean) / self.y_scale).astype(np.float32, copy=False)

        seed = int(kwargs.get("seed", self.config.get("seed", 7)))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        self._model = self._build_model()
        learning_rate = float(kwargs.get("learning_rate", self.config.get("learning_rate", 1e-2)))
        weight_decay = float(kwargs.get("weight_decay", self.config.get("weight_decay", 1e-6)))
        batch_size = int(kwargs.get("batch_size", self.config.get("batch_size", 512)))
        eval_batch_size = int(kwargs.get("eval_batch_size", self.config.get("eval_batch_size", batch_size)))
        num_epochs = int(kwargs.get("num_epochs", self.config.get("num_epochs", 100)))
        patience = int(kwargs.get("patience", self.config.get("patience", 12)))
        min_delta = float(kwargs.get("min_delta", self.config.get("min_delta", 1e-5)))
        clip_grad_norm = float(kwargs.get("clip_grad_norm", self.config.get("clip_grad_norm", 1.0)))
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_state: Optional[dict[str, torch.Tensor]] = None
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_since_improvement = 0

        for epoch in range(num_epochs):
            self._model.train()
            running_loss = 0.0
            seen = 0
            for batch_indices in _iter_batches(train_x_scaled.shape[0], batch_size, True, rng):
                xb = torch.as_tensor(
                    train_x_scaled[batch_indices],
                    device=self.runtime.device,
                    dtype=self.runtime.dtype,
                )
                yb = torch.as_tensor(
                    train_y_scaled[batch_indices],
                    device=self.runtime.device,
                    dtype=self.runtime.dtype,
                )
                optimizer.zero_grad(set_to_none=True)
                pred = self._model(xb)
                loss = torch.mean(torch.square(pred - yb))
                loss.backward()
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=clip_grad_norm)
                optimizer.step()
                batch_size_actual = int(batch_indices.shape[0])
                running_loss += float(loss.detach().cpu().item()) * batch_size_actual
                seen += batch_size_actual

            train_loss = running_loss / max(seen, 1)
            val_loss = self._evaluate_loss(val_x_scaled, val_y_scaled, batch_size=eval_batch_size)
            improved = val_loss < (best_val_loss - min_delta)
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_since_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self._model.state_dict().items()
                }
            else:
                epochs_since_improvement += 1
            if patience > 0 and epochs_since_improvement >= patience:
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        train_pred = self._predict_flat_raw(train_x_scaled, batch_size=eval_batch_size)
        val_pred = self._predict_flat_raw(val_x_scaled, batch_size=eval_batch_size)
        train_y_raw = train_y.astype(np.float32, copy=False)
        val_y_raw = val_y.astype(np.float32, copy=False)

        self.is_fitted = True
        self._last_kernel_recovery = None
        self.dataset_metadata = {
            "dataset_name": bundle.meta.dataset_name,
            "task_family": bundle.meta.task_family.value,
            "input_dim": bundle.meta.input_dim,
            "output_dim": bundle.meta.output_dim,
            "window_length": bundle.meta.window_length,
            "horizon": bundle.meta.horizon,
            "split_protocol": bundle.meta.split_protocol,
            "has_ground_truth_kernel": bundle.meta.has_ground_truth_kernel,
            "has_ground_truth_gfrf": bundle.meta.has_ground_truth_gfrf,
            "artifacts": bundle.artifacts.to_dict(),
            "extras": bundle.meta.extras,
        }
        self.training_summary = {
            "train_loss_scaled": float(train_loss),
            "val_loss_scaled": float(best_val_loss),
            "train_mse": _mse_numpy(train_y_raw, train_pred),
            "val_mse": _mse_numpy(val_y_raw, val_pred),
            "best_epoch": int(best_epoch),
            "epochs_ran": int(epoch + 1),
            "device_type": self.runtime.device_type,
            "dtype": self.runtime.dtype_name,
            "feature_size": int(self.feature_size),
            "output_size": int(self.output_size),
            "max_order": int(self.max_order),
            "order_ranks": {str(order): int(rank) for order, rank in sorted(self.order_ranks.items())},
            "num_parameters": int(sum(param.numel() for param in self._model.parameters())),
            "seed": int(seed),
        }

        return MethodResult(
            predictions={
                "val": val_pred.reshape((val_pred.shape[0],) + self.output_shape),
            },
            model_state_path=self.model_state_path,
            training_summary=dict(self.training_summary),
            kernel_recovery=None,
            metadata={
                "dataset_name": bundle.meta.dataset_name,
                "supports_kernel_recovery": True,
            },
        )

    def predict(self, X: Any, **kwargs: Any) -> Any:
        return self._predict_split(X, batch_size=kwargs.get("batch_size"))

    def save(self, path: PathLike) -> Path:
        self._require_fitted()
        target = Path(path).expanduser()
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            json_path = target
            weights_path = target.with_suffix(".npz")
        else:
            target.mkdir(parents=True, exist_ok=True)
            json_path = target / "method_state.json"
            weights_path = target / "model_weights.npz"

        payload = {
            "method_name": self.method_name,
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__,
            "config": dict(self.config),
            "runtime": self.runtime.to_dict(),
            "training_summary": dict(self.training_summary),
            "is_fitted": self.is_fitted,
            "state": {
                **self.get_state(),
                "weights_file": weights_path.name,
            },
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        np.savez_compressed(weights_path, **self._serialize_weight_arrays())
        self.model_state_path = str(json_path.resolve())
        return json_path

    def get_state(self) -> Mapping[str, Any]:
        return {
            "max_order": int(self.max_order),
            "order_ranks": {str(order): int(rank) for order, rank in sorted(self.order_ranks.items())},
            "window_length": self.window_length,
            "input_dim": self.input_dim,
            "feature_size": int(self.feature_size),
            "output_shape": list(self.output_shape),
            "output_size": int(self.output_size),
            "dataset_metadata": dict(self.dataset_metadata),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        self.max_order = int(state.get("max_order", 0))
        order_ranks_payload = state.get("order_ranks", {})
        if isinstance(order_ranks_payload, Mapping):
            self.order_ranks = {
                int(order): int(rank)
                for order, rank in order_ranks_payload.items()
                if int(rank) > 0
            }
        else:
            self.order_ranks = {}
        self.window_length = int(state["window_length"]) if state.get("window_length") is not None else None
        self.input_dim = int(state["input_dim"]) if state.get("input_dim") is not None else None
        self.feature_size = int(state.get("feature_size", 0))
        output_shape = state.get("output_shape", [])
        if isinstance(output_shape, Sequence):
            self.output_shape = tuple(int(dim) for dim in output_shape)
        else:
            self.output_shape = ()
        self.output_size = int(state.get("output_size", _flatten_output_shape(self.output_shape or (1,))))
        dataset_metadata = state.get("dataset_metadata", {})
        self.dataset_metadata = dict(dataset_metadata) if isinstance(dataset_metadata, Mapping) else {}
        self._last_kernel_recovery = None

        weights_file = state.get("weights_file")
        if weights_file is None:
            self._model = None
            return
        if self.model_state_path is None:
            raise RuntimeError("model_state_path is required to restore CP-Volterra weights.")
        weights_path = Path(self.model_state_path).with_name(str(weights_file))
        self._load_weight_arrays(weights_path)

    def _raw_cp_parameters(self) -> dict[str, Any]:
        self._require_fitted()
        assert self._model is not None
        assert self.x_scale is not None and self.y_scale is not None and self.y_mean is not None
        bias = self._model.bias.detach().cpu().numpy().astype(np.float32, copy=False)
        bias_raw = self.y_mean + self.y_scale * bias
        orders: list[dict[str, Any]] = []
        y_scale = self.y_scale.reshape(-1, 1)
        x_scale = self.x_scale.reshape(1, -1)

        for order in sorted(self.order_ranks):
            output_key = _CPVolterraModule._output_key(order)
            output_weights = self._model.output_weights[output_key].detach().cpu().numpy().astype(np.float32, copy=False)
            output_weights_raw = output_weights * y_scale
            factors_standardized: list[np.ndarray] = []
            factors_raw: list[np.ndarray] = []
            for mode in range(order):
                factor_key = _CPVolterraModule._factor_key(order, mode)
                factor = self._model.input_factors[factor_key].detach().cpu().numpy().astype(np.float32, copy=False)
                factors_standardized.append(factor.copy())
                factors_raw.append((factor / x_scale).astype(np.float32, copy=False))

            orders.append(
                {
                    "order": int(order),
                    "rank": int(self.order_ranks[order]),
                    "output_weights_standardized": output_weights.reshape(self.output_shape + (self.order_ranks[order],)),
                    "output_weights_raw": output_weights_raw.reshape(self.output_shape + (self.order_ranks[order],)),
                    "input_factors_standardized": [
                        factor.reshape((self.order_ranks[order], self.window_length, self.input_dim))
                        for factor in factors_standardized
                    ],
                    "input_factors_raw": [
                        factor.reshape((self.order_ranks[order], self.window_length, self.input_dim))
                        for factor in factors_raw
                    ],
                }
            )

        return {
            "representation": "cp_volterra",
            "bias_standardized": bias.reshape(self.output_shape),
            "bias_raw": bias_raw.reshape(self.output_shape),
            "orders": orders,
        }

    def _maybe_expand_full_kernel(
        self,
        *,
        order: int,
        output_weights_raw: np.ndarray,
        factors_raw: Sequence[np.ndarray],
        max_full_kernel_elements: int,
    ) -> tuple[Optional[np.ndarray], int]:
        full_shape_flat = (self.output_size,) + (self.feature_size,) * order
        element_count = _flatten_output_shape(full_shape_flat)
        if element_count > max_full_kernel_elements:
            return None, int(element_count)

        kernel = np.zeros(full_shape_flat, dtype=np.float32)
        rank = output_weights_raw.shape[1]
        for component in range(rank):
            component_kernel = output_weights_raw[:, component]
            for mode in range(order):
                component_kernel = np.multiply.outer(component_kernel, factors_raw[mode][component])
            kernel += component_kernel.astype(np.float32, copy=False)
        reshaped = kernel.reshape(self.output_shape + (self.window_length, self.input_dim) * order)
        return reshaped, int(element_count)

    def recover_kernels(self, **kwargs: Any) -> KernelRecoveryResult:
        self._require_fitted()
        expand_full = bool(kwargs.get("expand_full", False))
        max_full_kernel_elements = int(
            kwargs.get("max_full_kernel_elements", self.config.get("max_full_kernel_elements", _DEFAULT_KERNEL_EXPANSION_LIMIT))
        )
        cp_params = self._raw_cp_parameters()

        recovered_orders: list[dict[str, Any]] = []
        materialized_orders: list[int] = []
        skipped_orders: dict[int, int] = {}
        for order_payload in cp_params["orders"]:
            order = int(order_payload["order"])
            rank = int(order_payload["rank"])
            output_weights_raw = np.asarray(order_payload["output_weights_raw"], dtype=np.float32).reshape(self.output_size, rank)
            input_factors_raw = [
                np.asarray(factor, dtype=np.float32).reshape(rank, self.feature_size)
                for factor in order_payload["input_factors_raw"]
            ]
            full_kernel = None
            element_count = int(self.output_size * (self.feature_size ** order))
            if expand_full:
                full_kernel, element_count = self._maybe_expand_full_kernel(
                    order=order,
                    output_weights_raw=output_weights_raw,
                    factors_raw=input_factors_raw,
                    max_full_kernel_elements=max_full_kernel_elements,
                )
                if full_kernel is not None:
                    materialized_orders.append(order)
                else:
                    skipped_orders[order] = element_count
            recovered_orders.append(
                {
                    **order_payload,
                    "full_kernel": full_kernel,
                    "full_kernel_shape": self.output_shape + (self.window_length, self.input_dim) * order,
                    "materialized_full_kernel": full_kernel is not None,
                    "full_kernel_element_count": int(element_count),
                }
            )

        result = KernelRecoveryResult(
            kernels={
                "representation": "cp_volterra",
                "input_layout": {
                    "window_length": int(self.window_length or 0),
                    "input_dim": int(self.input_dim or 0),
                    "feature_size": int(self.feature_size),
                },
                "output_layout": {
                    "output_shape": list(self.output_shape),
                    "output_size": int(self.output_size),
                },
                "bias_standardized": cp_params["bias_standardized"],
                "bias_raw": cp_params["bias_raw"],
                "orders": recovered_orders,
            },
            summary={
                "supports_full_kernel_materialization": True,
                "expand_full_requested": expand_full,
                "max_full_kernel_elements": int(max_full_kernel_elements),
                "materialized_orders": materialized_orders,
                "skipped_orders": {str(order): count for order, count in sorted(skipped_orders.items())},
            },
            artifacts={},
        )
        self._last_kernel_recovery = result
        return result

    def _export_cp_parameter_npz(self, path: Path) -> None:
        kernels = self.recover_kernels(expand_full=False).kernels
        orders = kernels["orders"]
        payload: Dict[str, np.ndarray] = {
            "bias_standardized": np.asarray(kernels["bias_standardized"], dtype=np.float32),
            "bias_raw": np.asarray(kernels["bias_raw"], dtype=np.float32),
        }
        for order_payload in orders:
            order = int(order_payload["order"])
            payload[f"order_{order}_output_weights_standardized"] = np.asarray(
                order_payload["output_weights_standardized"],
                dtype=np.float32,
            )
            payload[f"order_{order}_output_weights_raw"] = np.asarray(
                order_payload["output_weights_raw"],
                dtype=np.float32,
            )
            for mode, factor in enumerate(order_payload["input_factors_standardized"]):
                payload[f"order_{order}_factor_{mode}_standardized"] = np.asarray(factor, dtype=np.float32)
            for mode, factor in enumerate(order_payload["input_factors_raw"]):
                payload[f"order_{order}_factor_{mode}_raw"] = np.asarray(factor, dtype=np.float32)
        np.savez_compressed(path, **payload)

    def export_artifacts(self, output_dir: PathLike) -> Mapping[str, Any]:
        self._require_fitted()
        root = Path(output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        state_path = self.save(root / "method_state.json")
        summary_path = root / "training_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(self.training_summary, handle, indent=2)

        kernel_summary = self.recover_kernels(expand_full=False)
        kernel_summary_path = root / "kernel_recovery_summary.json"
        with kernel_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(kernel_summary.summary, handle, indent=2)

        cp_params_path = root / "cp_parameters.npz"
        self._export_cp_parameter_npz(cp_params_path)

        artifacts: dict[str, ArtifactRef] = {
            "model_state": ArtifactRef(kind="method_state", path=str(state_path)),
            "training_summary": ArtifactRef(kind="training_summary", path=str(summary_path)),
            "kernel_recovery_summary": ArtifactRef(kind="kernel_recovery_summary", path=str(kernel_summary_path)),
            "cp_parameters": ArtifactRef(kind="cp_parameters", path=str(cp_params_path)),
        }

        export_full = bool(self.config.get("export_full_kernels", False))
        if export_full:
            kernels = self.recover_kernels(expand_full=True)
            for order_payload in kernels.kernels["orders"]:
                if order_payload["full_kernel"] is None:
                    continue
                order = int(order_payload["order"])
                kernel_path = root / f"kernel_order_{order}.npy"
                np.save(kernel_path, np.asarray(order_payload["full_kernel"], dtype=np.float32))
                artifacts[f"kernel_order_{order}"] = ArtifactRef(kind="full_kernel", path=str(kernel_path))

        return artifacts
