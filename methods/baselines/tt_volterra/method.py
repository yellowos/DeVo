"""Tensor-train Volterra baseline for nonlinear benchmark tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    torch = None
    nn = None

from methods.base import ArtifactRef, BaseMethod, KernelRecoveryResult, MethodResult, register_method
from methods.utils.device import select_device


PathLike = str | Path
_TorchModuleBase = nn.Module if nn is not None else object


def _ensure_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover - current environment includes torch.
        raise RuntimeError("TTVolterraMethod requires PyTorch to be installed.")


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _normalize_input_array(
    value: Any,
    *,
    expected_shape: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    array = _as_float_array(value, name="X")
    if array.ndim == 1:
        array = array.reshape(1, -1)

    if expected_shape is not None:
        expected = tuple(int(dim) for dim in expected_shape)
        if array.ndim == len(expected):
            array = array.reshape((1,) + expected)
        if tuple(array.shape[1:]) != expected:
            raise ValueError(f"Expected X trailing shape {expected}, got {tuple(array.shape[1:])}.")
        flat = array.reshape(array.shape[0], -1)
        return flat, expected

    if array.ndim < 2:
        raise ValueError("X must have shape [N, ...].")
    input_shape = tuple(int(dim) for dim in array.shape[1:])
    flat = array.reshape(array.shape[0], -1)
    return flat, input_shape


def _normalize_target_array(
    value: Any,
    *,
    expected_shape: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    array = _as_float_array(value, name="Y")
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if expected_shape is not None:
        expected = tuple(int(dim) for dim in expected_shape)
        if array.ndim == len(expected):
            array = array.reshape((1,) + expected)
        if tuple(array.shape[1:]) != expected:
            raise ValueError(f"Expected Y trailing shape {expected}, got {tuple(array.shape[1:])}.")
        flat = array.reshape(array.shape[0], -1)
        return flat, expected

    if array.ndim < 2:
        raise ValueError("Y must have shape [N, ...].")
    output_shape = tuple(int(dim) for dim in array.shape[1:])
    flat = array.reshape(array.shape[0], -1)
    return flat, output_shape


def _index_map(shape: Sequence[int]) -> np.ndarray:
    dims = tuple(int(dim) for dim in shape)
    if not dims:
        return np.zeros((1, 0), dtype=np.int64)
    grid = np.indices(dims, dtype=np.int64)
    return grid.reshape(len(dims), -1).T


def _safe_rms_scale(array: np.ndarray, *, minimum: float) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(array), axis=0))
    scale = np.asarray(scale, dtype=np.float64)
    scale[scale < minimum] = 1.0
    return scale


def _num_elements(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class _TTVolterraOrderBlock(_TorchModuleBase):
    """One Volterra order represented as a tensor train."""

    def __init__(
        self,
        *,
        feature_dim: int,
        output_dim: int,
        order: int,
        ranks: Sequence[int],
        init_std: float,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.order = int(order)
        self.ranks = [int(rank) for rank in ranks]

        if self.order < 1:
            raise ValueError("Volterra order must be at least 1.")

        self.cores = nn.ParameterList()
        if self.order == 1:
            self.cores.append(nn.Parameter(torch.empty(self.feature_dim, self.output_dim)))
        else:
            if len(self.ranks) != self.order - 1:
                raise ValueError(
                    f"Order {self.order} requires {self.order - 1} TT ranks, got {len(self.ranks)}."
                )
            self.cores.append(nn.Parameter(torch.empty(self.feature_dim, self.ranks[0])))
            for idx in range(1, self.order - 1):
                self.cores.append(
                    nn.Parameter(torch.empty(self.ranks[idx - 1], self.feature_dim, self.ranks[idx]))
                )
            self.cores.append(nn.Parameter(torch.empty(self.ranks[-1], self.feature_dim, self.output_dim)))
        self.reset_parameters(init_std=init_std)

    def reset_parameters(self, *, init_std: float) -> None:
        for core in self.cores:
            nn.init.normal_(core, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == 1:
            return x @ self.cores[0]

        state = torch.einsum("bf,fr->br", x, self.cores[0])
        for core in self.cores[1:-1]:
            state = torch.einsum("br,bf,rfh->bh", state, x, core)
        return torch.einsum("br,bf,rfo->bo", state, x, self.cores[-1])

    def export_internal_cores(self) -> list[np.ndarray]:
        return [core.detach().cpu().numpy() for core in self.cores]

    def load_internal_cores(self, arrays: Sequence[np.ndarray], *, device: torch.device, dtype: torch.dtype) -> None:
        if len(arrays) != len(self.cores):
            raise ValueError(f"Order {self.order} expected {len(self.cores)} cores, got {len(arrays)}.")
        with torch.no_grad():
            for parameter, array in zip(self.cores, arrays):
                tensor = torch.as_tensor(np.asarray(array), device=device, dtype=dtype)
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"Core shape mismatch for order {self.order}: expected {tuple(parameter.shape)}, got {tuple(tensor.shape)}."
                    )
                parameter.copy_(tensor)

    def export_uniform_cores(self) -> list[np.ndarray]:
        cores = self.export_internal_cores()
        if self.order == 1:
            return [cores[0][None, :, :]]
        export: list[np.ndarray] = [cores[0][None, :, :]]
        export.extend(cores[1:])
        return export


class _TTVolterraNetwork(_TorchModuleBase):
    """Bias plus a sum of TT-parameterized Volterra orders."""

    def __init__(
        self,
        *,
        feature_dim: int,
        output_dim: int,
        orders: Sequence[int],
        ranks_by_order: Mapping[int, Sequence[int]],
        init_std: float,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.orders = [int(order) for order in orders]
        self.ranks_by_order = {int(order): [int(rank) for rank in ranks] for order, ranks in ranks_by_order.items()}

        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.blocks = nn.ModuleDict(
            {
                str(order): _TTVolterraOrderBlock(
                    feature_dim=self.feature_dim,
                    output_dim=self.output_dim,
                    order=order,
                    ranks=self.ranks_by_order.get(order, ()),
                    init_std=init_std,
                )
                for order in self.orders
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.bias.unsqueeze(0).expand(x.shape[0], -1)
        for order in self.orders:
            output = output + self.blocks[str(order)](x)
        return output

    def export_internal_state(self) -> dict[str, np.ndarray]:
        payload = {"bias": self.bias.detach().cpu().numpy()}
        for order in self.orders:
            block = self.blocks[str(order)]
            for index, core in enumerate(block.export_internal_cores()):
                payload[f"order_{order}_core_{index}"] = core
        return payload

    def load_internal_state(self, payload: Mapping[str, np.ndarray], *, device: torch.device, dtype: torch.dtype) -> None:
        bias = payload.get("bias")
        if bias is None:
            raise ValueError("Serialized TT-Volterra payload missing bias.")
        with torch.no_grad():
            self.bias.copy_(torch.as_tensor(np.asarray(bias), device=device, dtype=dtype))

        for order in self.orders:
            block = self.blocks[str(order)]
            arrays: list[np.ndarray] = []
            for index in range(len(block.cores)):
                key = f"order_{order}_core_{index}"
                if key not in payload:
                    raise ValueError(f"Serialized TT-Volterra payload missing {key}.")
                arrays.append(np.asarray(payload[key]))
            block.load_internal_cores(arrays, device=device, dtype=dtype)

    def export_uniform_order_cores(self, order: int) -> list[np.ndarray]:
        return self.blocks[str(order)].export_uniform_cores()


@register_method("tt_volterra")
class TTVolterraMethod(BaseMethod):
    """Tensor-train Volterra baseline with compact kernel recovery."""

    SUPPORTS_KERNEL_RECOVERY = True

    DEFAULT_CONFIG = {
        "max_order": 3,
        "tt_rank": 8,
        "tt_ranks_by_order": {},
        "epochs": 80,
        "batch_size": 256,
        "predict_batch_size": 1024,
        "learning_rate": 1e-2,
        "weight_decay": 1e-6,
        "patience": 12,
        "min_delta": 1e-6,
        "gradient_clip_norm": 5.0,
        "shuffle": True,
        "init_std": 5e-2,
        "input_rms_normalization": True,
        "output_rms_normalization": True,
        "min_scale": 1e-6,
        "random_seed": 0,
        "dense_kernel_max_elements": 250000,
    }

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        super().__init__(config=config, device=device, dtype=dtype)
        self._model: Optional[_TTVolterraNetwork] = None
        self._input_shape: tuple[int, ...] = ()
        self._output_shape: tuple[int, ...] = ()
        self._flat_feature_dim: int = 0
        self._flat_output_dim: int = 0
        self._orders: list[int] = []
        self._ranks_by_order: dict[int, list[int]] = {}
        self._feature_scale: Optional[np.ndarray] = None
        self._target_scale: Optional[np.ndarray] = None
        self._feature_index_map: Optional[np.ndarray] = None
        self._output_index_map: Optional[np.ndarray] = None
        self._dataset_name: Optional[str] = None
        self._history: list[dict[str, float]] = []

    def _effective_config(self, **overrides: Any) -> dict[str, Any]:
        payload = dict(self.DEFAULT_CONFIG)
        payload.update(self.config)
        for key, value in overrides.items():
            if value is not None:
                payload[key] = value
        return payload

    def _require_fitted_model(self) -> _TTVolterraNetwork:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("TTVolterraMethod has not been fitted.")
        return self._model

    def _resolve_orders_and_ranks(self, max_order: int, tt_rank: Any, by_order: Any) -> tuple[list[int], dict[int, list[int]]]:
        if max_order < 1:
            raise ValueError("max_order must be >= 1.")
        orders = list(range(1, int(max_order) + 1))
        ranks_by_order: dict[int, list[int]] = {}
        rank_lookup = dict(by_order) if isinstance(by_order, Mapping) else {}

        for order in orders:
            if order == 1:
                ranks_by_order[order] = []
                continue
            rank_spec = rank_lookup.get(order)
            if rank_spec is None:
                rank_spec = rank_lookup.get(str(order), tt_rank)
            if isinstance(rank_spec, Sequence) and not isinstance(rank_spec, (str, bytes)):
                ranks = [max(1, int(rank)) for rank in rank_spec]
            else:
                ranks = [max(1, int(rank_spec))] * (order - 1)
            if len(ranks) != order - 1:
                raise ValueError(f"Order {order} requires {order - 1} TT rank values, got {len(ranks)}.")
            ranks_by_order[order] = ranks
        return orders, ranks_by_order

    def _build_model(
        self,
        *,
        feature_dim: int,
        output_dim: int,
        orders: Sequence[int],
        ranks_by_order: Mapping[int, Sequence[int]],
        init_std: float,
    ) -> _TTVolterraNetwork:
        _ensure_torch()
        model = _TTVolterraNetwork(
            feature_dim=feature_dim,
            output_dim=output_dim,
            orders=orders,
            ranks_by_order=ranks_by_order,
            init_std=float(init_std),
        )
        return model.to(device=self.runtime.device, dtype=self.runtime.dtype)

    def _prepare_split_payload(
        self,
        *,
        X: Any,
        Y: Any,
        expected_input_shape: Optional[Sequence[int]] = None,
        expected_output_shape: Optional[Sequence[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
        x_flat, input_shape = _normalize_input_array(X, expected_shape=expected_input_shape)
        y_flat, output_shape = _normalize_target_array(Y, expected_shape=expected_output_shape)
        if x_flat.shape[0] != y_flat.shape[0]:
            raise ValueError(f"X/Y sample count mismatch: {x_flat.shape[0]} vs {y_flat.shape[0]}.")
        return x_flat, y_flat, input_shape, output_shape

    def _evaluate_loss(self, x: np.ndarray, y: np.ndarray, *, batch_size: int) -> float:
        model = self._require_fitted_model()
        if x.shape[0] == 0:
            return 0.0
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                stop = min(start + batch_size, x.shape[0])
                xb = torch.as_tensor(x[start:stop], device=self.runtime.device, dtype=self.runtime.dtype)
                yb = torch.as_tensor(y[start:stop], device=self.runtime.device, dtype=self.runtime.dtype)
                pred = model(xb)
                loss = torch.mean((pred - yb) ** 2)
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else 0.0

    def fit(self, dataset_bundle: Any, **kwargs: Any) -> MethodResult:
        _ensure_torch()
        bundle = self.normalize_dataset_bundle(dataset_bundle, dataset_name=kwargs.get("dataset_name"))
        config = self._effective_config(**kwargs)
        self.config = {key: config[key] for key in self.DEFAULT_CONFIG}

        train_x, train_y, input_shape, output_shape = self._prepare_split_payload(
            X=bundle.train.X,
            Y=bundle.train.Y,
        )
        val_x, val_y, _, _ = self._prepare_split_payload(
            X=bundle.val.X,
            Y=bundle.val.Y,
            expected_input_shape=input_shape,
            expected_output_shape=output_shape,
        )
        self._prepare_split_payload(
            X=bundle.test.X,
            Y=bundle.test.Y,
            expected_input_shape=input_shape,
            expected_output_shape=output_shape,
        )

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._flat_feature_dim = int(train_x.shape[1])
        self._flat_output_dim = int(train_y.shape[1])
        self._feature_index_map = _index_map(self._input_shape)
        self._output_index_map = _index_map(self._output_shape)
        self._dataset_name = bundle.meta.dataset_name

        orders, ranks_by_order = self._resolve_orders_and_ranks(
            int(config["max_order"]),
            config["tt_rank"],
            config.get("tt_ranks_by_order", {}),
        )
        self._orders = orders
        self._ranks_by_order = {order: list(ranks_by_order[order]) for order in orders}

        min_scale = float(config["min_scale"])
        self._feature_scale = (
            _safe_rms_scale(train_x, minimum=min_scale)
            if bool(config["input_rms_normalization"])
            else np.ones(self._flat_feature_dim, dtype=np.float64)
        )
        self._target_scale = (
            _safe_rms_scale(train_y, minimum=min_scale)
            if bool(config["output_rms_normalization"])
            else np.ones(self._flat_output_dim, dtype=np.float64)
        )

        train_x_scaled = train_x / self._feature_scale
        val_x_scaled = val_x / self._feature_scale
        train_y_scaled = train_y / self._target_scale
        val_y_scaled = val_y / self._target_scale

        seed = int(config["random_seed"])
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        self._model = self._build_model(
            feature_dim=self._flat_feature_dim,
            output_dim=self._flat_output_dim,
            orders=self._orders,
            ranks_by_order=self._ranks_by_order,
            init_std=float(config["init_std"]),
        )

        batch_size = max(1, int(config["batch_size"]))
        learning_rate = float(config["learning_rate"])
        epochs = max(1, int(config["epochs"]))
        weight_decay = float(config["weight_decay"])
        patience = max(0, int(config["patience"]))
        min_delta = float(config["min_delta"])
        gradient_clip = float(config["gradient_clip_norm"])

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_state: Optional[dict[str, torch.Tensor]] = None
        self._history = []
        eval_x = val_x_scaled if val_x_scaled.shape[0] else train_x_scaled
        eval_y = val_y_scaled if val_y_scaled.shape[0] else train_y_scaled

        for epoch in range(1, epochs + 1):
            self._model.train()
            if bool(config["shuffle"]):
                indices = rng.permutation(train_x_scaled.shape[0])
            else:
                indices = np.arange(train_x_scaled.shape[0])

            batch_losses: list[float] = []
            for start in range(0, train_x_scaled.shape[0], batch_size):
                stop = min(start + batch_size, train_x_scaled.shape[0])
                batch_idx = indices[start:stop]
                xb = torch.as_tensor(train_x_scaled[batch_idx], device=self.runtime.device, dtype=self.runtime.dtype)
                yb = torch.as_tensor(train_y_scaled[batch_idx], device=self.runtime.device, dtype=self.runtime.dtype)
                optimizer.zero_grad(set_to_none=True)
                pred = self._model(xb)
                loss = torch.mean((pred - yb) ** 2)
                loss.backward()
                if gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=gradient_clip)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            self.is_fitted = True
            val_loss = self._evaluate_loss(eval_x, eval_y, batch_size=max(batch_size, int(config["predict_batch_size"])))
            self._history.append(
                {
                    "epoch": int(epoch),
                    "train_mse": float(train_loss),
                    "val_mse": float(val_loss),
                }
            )

            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in self._model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if patience and epochs_without_improvement >= patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        train_mse = self._evaluate_loss(
            train_x_scaled,
            train_y_scaled,
            batch_size=max(batch_size, int(config["predict_batch_size"])),
        )
        final_val_mse = self._evaluate_loss(
            val_x_scaled,
            val_y_scaled,
            batch_size=max(batch_size, int(config["predict_batch_size"])),
        )

        self.training_summary = {
            "dataset_name": bundle.meta.dataset_name,
            "task_family": bundle.meta.task_family.value,
            "input_shape": list(self._input_shape),
            "output_shape": list(self._output_shape),
            "flat_feature_dim": self._flat_feature_dim,
            "flat_output_dim": self._flat_output_dim,
            "orders": list(self._orders),
            "ranks_by_order": {str(order): list(ranks) for order, ranks in self._ranks_by_order.items()},
            "parameter_count": int(sum(parameter.numel() for parameter in self._model.parameters())),
            "epochs_completed": len(self._history),
            "best_epoch": int(best_epoch),
            "train_mse": float(train_mse),
            "val_mse": float(final_val_mse),
            "used_validation_split": bool(val_x_scaled.shape[0]),
            "input_rms_normalization": bool(config["input_rms_normalization"]),
            "output_rms_normalization": bool(config["output_rms_normalization"]),
            "device": self.runtime.device_type,
            "dtype": self.runtime.dtype_name,
            "fallback_reason": self.runtime.fallback_reason,
            "history": [dict(item) for item in self._history],
        }

        save_path = kwargs.get("save_path")
        model_state_path: Optional[str] = None
        if save_path is not None:
            model_state_path = str(self.save(save_path))

        predictions: Any = None
        if bool(kwargs.get("return_split_predictions", False)):
            predictions = {
                "train": self.predict(bundle.train.X),
                "val": self.predict(bundle.val.X),
                "test": self.predict(bundle.test.X),
            }

        return MethodResult(
            predictions=predictions,
            model_state_path=model_state_path or self.model_state_path,
            training_summary=dict(self.training_summary),
            metadata={
                "method_name": self.method_name,
                "supports_kernel_recovery": True,
            },
        )

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        model = self._require_fitted_model()
        if self._feature_scale is None or self._target_scale is None:
            raise RuntimeError("TTVolterraMethod is missing normalization state.")

        batch_size = max(1, int(kwargs.get("batch_size", self.config.get("predict_batch_size", self.DEFAULT_CONFIG["predict_batch_size"]))))
        x_flat, _ = _normalize_input_array(X, expected_shape=self._input_shape)
        x_scaled = x_flat / self._feature_scale

        outputs: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, x_scaled.shape[0], batch_size):
                stop = min(start + batch_size, x_scaled.shape[0])
                xb = torch.as_tensor(x_scaled[start:stop], device=self.runtime.device, dtype=self.runtime.dtype)
                pred = model(xb).detach().cpu().numpy()
                outputs.append(pred)

        if not outputs:
            return np.zeros((0,) + self._output_shape, dtype=np.float64)

        stacked = np.concatenate(outputs, axis=0)
        restored = stacked * self._target_scale
        return restored.reshape((restored.shape[0],) + self._output_shape)

    def _serialized_state_manifest(self, *, weights_file: str) -> dict[str, Any]:
        return {
            "weights_file": weights_file,
            "input_shape": list(self._input_shape),
            "output_shape": list(self._output_shape),
            "flat_feature_dim": int(self._flat_feature_dim),
            "flat_output_dim": int(self._flat_output_dim),
            "orders": list(self._orders),
            "ranks_by_order": {str(order): list(ranks) for order, ranks in self._ranks_by_order.items()},
            "feature_scale_key": "feature_scale",
            "target_scale_key": "target_scale",
            "dataset_name": self._dataset_name,
        }

    def _weights_payload(self) -> dict[str, np.ndarray]:
        model = self._require_fitted_model()
        if self._feature_scale is None or self._target_scale is None:
            raise RuntimeError("TTVolterraMethod is missing normalization state.")
        payload = model.export_internal_state()
        payload["feature_scale"] = np.asarray(self._feature_scale, dtype=np.float64)
        payload["target_scale"] = np.asarray(self._target_scale, dtype=np.float64)
        return payload

    def save(self, path: PathLike) -> Path:
        model = self._require_fitted_model()
        del model
        target = Path(path).expanduser()
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            state_path = target
            bundle_dir = target.parent
        else:
            target.mkdir(parents=True, exist_ok=True)
            bundle_dir = target
            state_path = target / "method_state.json"

        weights_path = bundle_dir / "parameters.npz"
        np.savez_compressed(weights_path, **self._weights_payload())

        payload = {
            "method_name": self.method_name,
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__,
            "config": dict(self.config),
            "runtime": self.runtime.to_dict(),
            "training_summary": dict(self.training_summary),
            "is_fitted": self.is_fitted,
            "state": self._serialized_state_manifest(weights_file=weights_path.name),
        }
        _json_dump(state_path, payload)
        self.model_state_path = str(state_path.resolve())
        return state_path

    @classmethod
    def load(cls, path: PathLike) -> "TTVolterraMethod":
        source = Path(path).expanduser().resolve()
        state_path = source / "method_state.json" if source.is_dir() else source
        with state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Serialized TT-Volterra payload must be a JSON object: {state_path}")

        serialized_method_name = payload.get("method_name")
        expected_name = cls.METHOD_NAME or cls.__name__.lower()
        if isinstance(serialized_method_name, str) and serialized_method_name.strip():
            if cls._normalize_method_name(serialized_method_name) != cls._normalize_method_name(expected_name):
                raise ValueError(
                    f"Serialized state belongs to '{serialized_method_name}', not '{expected_name}'."
                )

        runtime_payload = payload.get("runtime", {})
        device = runtime_payload.get("device_type") if isinstance(runtime_payload, Mapping) else None
        dtype = runtime_payload.get("dtype") if isinstance(runtime_payload, Mapping) else None
        config = payload.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError("Serialized TT-Volterra config must be a mapping.")

        instance = cls(config=config, device=device, dtype=dtype)
        if isinstance(runtime_payload, Mapping):
            instance.runtime = select_device(
                preferred_device=runtime_payload.get("device_type"),
                preferred_dtype=runtime_payload.get("dtype"),
            )
        summary = payload.get("training_summary", {})
        if isinstance(summary, Mapping):
            instance.training_summary = dict(summary)
        instance.is_fitted = bool(payload.get("is_fitted", False))
        instance.model_state_path = str(state_path)
        state = payload.get("state", {})
        if not isinstance(state, Mapping):
            raise ValueError("Serialized TT-Volterra state must be a mapping.")
        instance._restore_serialized_state(state=state, base_dir=state_path.parent)
        return instance

    def _restore_serialized_state(self, *, state: Mapping[str, Any], base_dir: Path) -> None:
        _ensure_torch()
        input_shape = tuple(int(dim) for dim in state.get("input_shape", ()))
        output_shape = tuple(int(dim) for dim in state.get("output_shape", ()))
        if not input_shape or not output_shape:
            raise ValueError("Serialized TT-Volterra state is missing input/output shape metadata.")

        orders = [int(order) for order in state.get("orders", [])]
        if not orders:
            raise ValueError("Serialized TT-Volterra state is missing Volterra orders.")
        rank_payload = state.get("ranks_by_order", {})
        if not isinstance(rank_payload, Mapping):
            raise ValueError("Serialized TT-Volterra ranks_by_order must be a mapping.")
        ranks_by_order = {
            int(order): [int(rank) for rank in ranks]
            for order, ranks in rank_payload.items()
        }

        feature_dim = int(state.get("flat_feature_dim", _num_elements(input_shape)))
        output_dim = int(state.get("flat_output_dim", _num_elements(output_shape)))
        weights_file = state.get("weights_file")
        if not isinstance(weights_file, str) or not weights_file:
            raise ValueError("Serialized TT-Volterra state is missing weights_file.")

        weights_path = base_dir / weights_file
        with np.load(weights_path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}

        if "feature_scale" not in payload or "target_scale" not in payload:
            raise ValueError("Serialized TT-Volterra weights are missing normalization arrays.")

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._flat_feature_dim = feature_dim
        self._flat_output_dim = output_dim
        self._orders = orders
        self._ranks_by_order = {int(order): list(ranks_by_order.get(int(order), ())) for order in orders}
        self._feature_scale = np.asarray(payload["feature_scale"], dtype=np.float64)
        self._target_scale = np.asarray(payload["target_scale"], dtype=np.float64)
        self._feature_index_map = _index_map(self._input_shape)
        self._output_index_map = _index_map(self._output_shape)
        dataset_name = state.get("dataset_name")
        self._dataset_name = str(dataset_name) if dataset_name else None

        init_std = float(self.config.get("init_std", self.DEFAULT_CONFIG["init_std"]))
        self._model = self._build_model(
            feature_dim=self._flat_feature_dim,
            output_dim=self._flat_output_dim,
            orders=self._orders,
            ranks_by_order=self._ranks_by_order,
            init_std=init_std,
        )
        self._model.load_internal_state(payload, device=self.runtime.device, dtype=self.runtime.dtype)

    def _order_element_count(self, order: int) -> int:
        return int((self._flat_feature_dim ** int(order)) * self._flat_output_dim)

    def _structured_kernel_shape(self, order: int) -> tuple[int, ...]:
        return tuple(self._input_shape) * int(order) + tuple(self._output_shape)

    def _export_order_compact_kernel(self, order: int) -> dict[str, Any]:
        model = self._require_fitted_model()
        if self._feature_scale is None or self._target_scale is None:
            raise RuntimeError("TTVolterraMethod is missing normalization state.")

        uniform_cores = model.export_uniform_order_cores(order)
        scaled_cores: list[np.ndarray] = []
        for core in uniform_cores:
            core_copy = np.asarray(core, dtype=np.float64).copy()
            core_copy = core_copy / self._feature_scale[None, :, None]
            scaled_cores.append(core_copy)
        scaled_cores[-1] = scaled_cores[-1] * self._target_scale[None, None, :]

        return {
            "format": "tensor_train",
            "coordinate_system": "original_data_space",
            "mode_sizes": [int(self._flat_feature_dim)] * int(order),
            "output_dim": int(self._flat_output_dim),
            "ranks": [int(core.shape[-1]) for core in scaled_cores[:-1]],
            "cores": scaled_cores,
        }

    def materialize_kernel(
        self,
        order: int,
        *,
        reshape_modes: bool = True,
        max_elements: Optional[int] = None,
    ) -> np.ndarray:
        if int(order) not in self._orders:
            raise KeyError(f"Order {order} is not present in this TT-Volterra model.")
        compact = self._export_order_compact_kernel(int(order))
        dense_shape = [self._flat_feature_dim] * int(order) + [self._flat_output_dim]
        element_count = self._order_element_count(int(order))
        if max_elements is not None and element_count > int(max_elements):
            raise MemoryError(
                f"Dense order-{order} kernel requires {element_count} elements, above limit {int(max_elements)}."
            )

        dense = np.asarray(compact["cores"][0], dtype=np.float64)
        for core in compact["cores"][1:]:
            dense = np.tensordot(dense, np.asarray(core, dtype=np.float64), axes=([-1], [0]))
        dense = np.asarray(dense.squeeze(0), dtype=np.float64).reshape(dense_shape)
        if not reshape_modes:
            return dense
        return dense.reshape(self._structured_kernel_shape(int(order)))

    def recover_kernels(self, **kwargs: Any) -> KernelRecoveryResult:
        self._require_fitted_model()
        max_dense_elements = kwargs.get(
            "max_dense_elements",
            self.config.get("dense_kernel_max_elements", self.DEFAULT_CONFIG["dense_kernel_max_elements"]),
        )
        dense_preference = kwargs.get("as_dense")

        if self._feature_index_map is None or self._output_index_map is None:
            raise RuntimeError("TTVolterraMethod is missing index maps.")

        model = self._require_fitted_model()
        bias = np.asarray(model.bias.detach().cpu().numpy(), dtype=np.float64)
        if self._target_scale is None:
            raise RuntimeError("TTVolterraMethod is missing target normalization state.")
        bias = (bias * self._target_scale).reshape(self._output_shape)

        orders_payload: dict[str, Any] = {}
        dense_orders: list[int] = []
        compact_only_orders: list[int] = []
        for order in self._orders:
            element_count = self._order_element_count(order)
            should_materialize = False
            if dense_preference is True:
                should_materialize = True
            elif dense_preference is False:
                should_materialize = False
            else:
                should_materialize = element_count <= int(max_dense_elements)

            compact = self._export_order_compact_kernel(order)
            dense_kernel: Optional[np.ndarray] = None
            if should_materialize:
                dense_kernel = self.materialize_kernel(order, reshape_modes=True, max_elements=int(max_dense_elements))
                dense_orders.append(int(order))
            else:
                compact_only_orders.append(int(order))

            orders_payload[str(order)] = {
                "order": int(order),
                "element_count": int(element_count),
                "flat_shape": [int(self._flat_feature_dim)] * int(order) + [int(self._flat_output_dim)],
                "structured_shape": list(self._structured_kernel_shape(order)),
                "compact": compact,
                "dense": dense_kernel,
                "dense_materialized": dense_kernel is not None,
            }

        kernels = {
            "representation": "tt_volterra",
            "coordinate_system": "original_data_space",
            "bias": bias,
            "input_shape": list(self._input_shape),
            "output_shape": list(self._output_shape),
            "flat_feature_dim": int(self._flat_feature_dim),
            "flat_output_dim": int(self._flat_output_dim),
            "feature_index_map": np.asarray(self._feature_index_map, dtype=np.int64),
            "output_index_map": np.asarray(self._output_index_map, dtype=np.int64),
            "flattening": {
                "input": "C-order flatten over trailing X dimensions; for [M, D], feature_index = lag * D + channel.",
                "output": "C-order flatten over trailing Y dimensions before training, then reshape on export.",
            },
            "orders": orders_payload,
            "on_demand_dense_api": "materialize_kernel(order, reshape_modes=True, max_elements=...)",
        }
        summary = {
            "dataset_name": self._dataset_name,
            "orders": list(self._orders),
            "dense_orders": dense_orders,
            "compact_only_orders": compact_only_orders,
            "dense_kernel_max_elements": int(max_dense_elements),
            "structured_dense_shape_convention": "[input_shape] repeated by order, followed by output_shape.",
            "symmetry_enforced": False,
        }
        return KernelRecoveryResult(kernels=kernels, summary=summary)

    def export_artifacts(self, output_dir: PathLike) -> Mapping[str, Any]:
        self._require_fitted_model()
        target = Path(output_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)

        model_state_path = self.save(target / "method_state.json")
        summary_path = target / "training_summary.json"
        _json_dump(summary_path, dict(self.training_summary))

        recovery = self.recover_kernels()
        kernels = recovery.kernels
        kernel_dir = target / "kernels"
        kernel_dir.mkdir(parents=True, exist_ok=True)

        bias_path = kernel_dir / "bias.npy"
        np.save(bias_path, np.asarray(kernels["bias"], dtype=np.float64))

        feature_index_map_path = kernel_dir / "feature_index_map.npy"
        output_index_map_path = kernel_dir / "output_index_map.npy"
        np.save(feature_index_map_path, np.asarray(kernels["feature_index_map"], dtype=np.int64))
        np.save(output_index_map_path, np.asarray(kernels["output_index_map"], dtype=np.int64))

        kernel_manifest_orders: dict[str, Any] = {}
        for order_key, payload in kernels["orders"].items():
            order = int(order_key)
            compact = payload["compact"]
            compact_path = kernel_dir / f"order_{order}_tt_cores.npz"
            np.savez_compressed(
                compact_path,
                **{f"core_{index}": np.asarray(core, dtype=np.float64) for index, core in enumerate(compact["cores"])},
            )

            dense_path: Optional[Path] = None
            if payload["dense"] is not None:
                dense_path = kernel_dir / f"order_{order}_dense.npy"
                np.save(dense_path, np.asarray(payload["dense"], dtype=np.float64))

            kernel_manifest_orders[order_key] = {
                "order": order,
                "format": compact["format"],
                "coordinate_system": compact["coordinate_system"],
                "mode_sizes": [int(size) for size in compact["mode_sizes"]],
                "output_dim": int(compact["output_dim"]),
                "ranks": [int(rank) for rank in compact["ranks"]],
                "compact_cores_file": str(compact_path),
                "dense_file": str(dense_path) if dense_path is not None else None,
                "structured_shape": [int(dim) for dim in payload["structured_shape"]],
                "dense_materialized": bool(payload["dense_materialized"]),
            }

        kernel_manifest_path = target / "kernel_recovery_manifest.json"
        _json_dump(
            kernel_manifest_path,
            {
                "representation": kernels["representation"],
                "coordinate_system": kernels["coordinate_system"],
                "input_shape": [int(dim) for dim in kernels["input_shape"]],
                "output_shape": [int(dim) for dim in kernels["output_shape"]],
                "flat_feature_dim": int(kernels["flat_feature_dim"]),
                "flat_output_dim": int(kernels["flat_output_dim"]),
                "flattening": dict(kernels["flattening"]),
                "bias_file": str(bias_path),
                "feature_index_map_file": str(feature_index_map_path),
                "output_index_map_file": str(output_index_map_path),
                "orders": kernel_manifest_orders,
                "summary": dict(recovery.summary),
                "on_demand_dense_api": kernels["on_demand_dense_api"],
            },
        )

        return {
            "model_state": ArtifactRef(kind="json", path=str(model_state_path)).to_dict(),
            "training_summary": ArtifactRef(kind="json", path=str(summary_path)).to_dict(),
            "kernel_manifest": ArtifactRef(kind="json", path=str(kernel_manifest_path)).to_dict(),
            "kernel_directory": ArtifactRef(kind="directory", path=str(kernel_dir)).to_dict(),
        }
