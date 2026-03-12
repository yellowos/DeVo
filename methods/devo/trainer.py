"""Training and method wrapper for DeVo."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.adapters.base import DatasetMeta
from methods.base import BaseMethod, MethodResult, register_method
from methods.utils import coerce_dataset_bundle, set_random_seed

from .attribution import prediction_error_gradient_attribution
from .model import DeVoConfig, DeVoModel
from .recovery import RecoveredKernelBundle, recover_devo_kernels


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_input_array(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError("Expected X with shape [N, M, D] or [N, M].")
    return np.asarray(array, dtype=np.float32)


def _prepare_target_array(value: Any, *, horizon: int, output_dim: int) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 1:
        if horizon != 1 or output_dim != 1:
            raise ValueError("1D Y only supports horizon=1 and output_dim=1.")
        array = array[:, None, None]
    elif array.ndim == 2:
        if array.shape[1] == output_dim and horizon == 1:
            array = array[:, None, :]
        elif array.shape[1] == horizon and output_dim == 1:
            array = array[:, :, None]
        elif array.shape[1] == horizon * output_dim:
            array = array.reshape(array.shape[0], horizon, output_dim)
        else:
            raise ValueError(
                "2D Y must be [N, output_dim], [N, horizon], or [N, horizon * output_dim]."
            )
    elif array.ndim != 3:
        raise ValueError("Expected Y with shape [N, H, O], [N, O], [N, H], or [N].")

    if array.shape[1] != horizon or array.shape[2] != output_dim:
        raise ValueError(
            f"Expected Y shape [N, {horizon}, {output_dim}], got {tuple(array.shape)}."
        )
    return np.asarray(array, dtype=np.float32)


def _to_loader(X: np.ndarray, Y: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def _dtype_name(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("torch."):
        return text.split(".", 1)[1]
    return text or "float32"


def _config_to_mapping(config: DeVoConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["orders"] = list(config.orders)
    payload["dtype"] = _dtype_name(config.dtype)
    return payload


def _config_from_mapping(value: Mapping[str, Any]) -> DeVoConfig:
    payload = dict(value)
    payload["orders"] = tuple(int(order) for order in payload.get("orders", (1, 2, 3)))
    dtype_name = str(payload.get("dtype", "float32"))
    payload["dtype"] = getattr(torch, dtype_name, torch.float32)
    return DeVoConfig(**payload)


def _weights_path(target: Path) -> Path:
    return target.with_suffix(".pt")


@register_method("devo")
class DeVoMethod(BaseMethod):
    """High-level DeVo method wrapper."""

    def __init__(self, config: Optional[DeVoConfig] = None, **config_overrides: Any) -> None:
        self.devo_config = config or DeVoConfig(**config_overrides)
        super().__init__(
            config=_config_to_mapping(self.devo_config),
            device=self.devo_config.device,
            dtype=self.devo_config.dtype,
        )
        self.model: Optional[DeVoModel] = None
        self.training_history: list[Dict[str, float]] = []
        self._model_spec: Optional[Dict[str, int]] = None

    @classmethod
    def _from_serialized_payload(cls, payload: Mapping[str, Any]) -> "DeVoMethod":
        config_payload = payload.get("config", {})
        if not isinstance(config_payload, Mapping):
            raise ValueError("Serialized DeVo config must be a mapping.")
        return cls(config=_config_from_mapping(config_payload))

    def _build_model(self, *, window_length: int, input_dim: int, output_dim: int, horizon: int) -> DeVoModel:
        return DeVoModel(
            window_length=window_length,
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            config=self.devo_config,
        ).to(device=self.device, dtype=self.dtype)

    def _evaluate_loss(self, loader: DataLoader) -> float:
        assert self.model is not None
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)
                pred = self.model(x_batch)
                loss = torch.mean((pred - y_batch) ** 2)
                total += float(loss.item()) * x_batch.shape[0]
                count += x_batch.shape[0]
        return total / max(count, 1)

    def fit(self, dataset_bundle: Any, **kwargs: Any) -> MethodResult:
        bundle = coerce_dataset_bundle(dataset_bundle)
        self.bundle_meta = bundle.meta
        self._model_spec = {
            "window_length": int(bundle.meta.window_length),
            "input_dim": int(bundle.meta.input_dim),
            "output_dim": int(bundle.meta.output_dim),
            "horizon": int(bundle.meta.horizon),
        }
        set_random_seed(self.devo_config.seed)

        X_train = _prepare_input_array(bundle.train.X)
        Y_train = _prepare_target_array(
            bundle.train.Y,
            horizon=bundle.meta.horizon,
            output_dim=bundle.meta.output_dim,
        )
        X_val = _prepare_input_array(bundle.val.X)
        Y_val = _prepare_target_array(
            bundle.val.Y,
            horizon=bundle.meta.horizon,
            output_dim=bundle.meta.output_dim,
        )

        self.model = self._build_model(**self._model_spec)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.devo_config.learning_rate,
            weight_decay=self.devo_config.weight_decay,
        )

        train_loader = _to_loader(
            X_train,
            Y_train,
            batch_size=int(kwargs.get("batch_size", self.devo_config.batch_size)),
            shuffle=True,
            seed=self.devo_config.seed,
        )
        val_loader = _to_loader(
            X_val,
            Y_val,
            batch_size=int(kwargs.get("eval_batch_size", self.devo_config.eval_batch_size)),
            shuffle=False,
            seed=self.devo_config.seed,
        )

        epochs = int(kwargs.get("epochs", self.devo_config.epochs))
        verbose = bool(kwargs.get("verbose", self.devo_config.verbose))
        self.training_history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            count = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(x_batch)
                loss = torch.mean((pred - y_batch) ** 2)
                loss.backward()
                if self.devo_config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.devo_config.grad_clip_norm)
                optimizer.step()
                total += float(loss.item()) * x_batch.shape[0]
                count += x_batch.shape[0]

            train_loss = total / max(count, 1)
            val_loss = self._evaluate_loss(val_loader)
            record = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            self.training_history.append(record)

            if verbose and (epoch % max(self.devo_config.log_every, 1) == 0 or epoch == epochs):
                device_label = getattr(self.device, "type", str(self.device))
                print(
                    f"[DeVo] epoch={epoch:03d} train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} device={device_label}"
                )

        self.training_summary = {
            "dataset_name": bundle.meta.dataset_name,
            "train_samples": int(bundle.train.num_samples),
            "val_samples": int(bundle.val.num_samples),
            "epochs": epochs,
            "final_train_loss": float(self.training_history[-1]["train_loss"]),
            "final_val_loss": float(self.training_history[-1]["val_loss"]),
            "device_type": self.runtime.device_type,
            "dtype": self.runtime.dtype_name,
        }
        self.is_fitted = True
        return MethodResult(
            predictions=None,
            model_state_path=self.model_state_path,
            training_summary=dict(self.training_summary),
            artifacts={},
            metadata={"method_name": self.method_name},
        )

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() before predict().")

        array = _prepare_input_array(X)
        batch_size = int(kwargs.get("batch_size", self.devo_config.eval_batch_size))
        self.model.eval()
        outputs = []

        with torch.no_grad():
            for start in range(0, array.shape[0], batch_size):
                stop = min(start + batch_size, array.shape[0])
                batch = torch.from_numpy(array[start:stop]).to(self.device, dtype=self.dtype)
                pred = self.model(batch).detach().cpu().numpy()
                outputs.append(pred)

        return np.concatenate(outputs, axis=0)

    def supports_kernel_recovery(self) -> bool:
        return True

    def recover_kernels(self, **kwargs: Any) -> RecoveredKernelBundle:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() before recover_kernels().")
        return recover_devo_kernels(self.model, **kwargs)

    def export_parameters(self) -> Dict[str, object]:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() before export_parameters().")
        return self.model.export_parameters()

    def export_artifacts(self, output_dir: str | Path) -> Mapping[str, Any]:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() before export_artifacts().")

        target = Path(output_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        state_path = target / "devo_model_state.pt"
        params_path = target / "devo_parameters.pt"
        torch.save(self.model.state_dict(), state_path)
        torch.save(self.model.export_parameters(), params_path)
        return {
            "model_state": str(state_path),
            "parameters": str(params_path),
        }

    def attribute(self, X: Any, Y: Optional[Any] = None, **kwargs: Any) -> np.ndarray:
        """Return a generic `[sample, time, variable]` attribution tensor."""

        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() before attribute().")

        inputs = torch.from_numpy(_prepare_input_array(X))
        targets = None
        if Y is not None:
            assert self.bundle_meta is not None
            targets = torch.from_numpy(
                _prepare_target_array(
                    Y,
                    horizon=self.bundle_meta.horizon,
                    output_dim=self.bundle_meta.output_dim,
                )
            )
        attribution = prediction_error_gradient_attribution(
            self.model,
            inputs,
            targets,
            batch_size=int(kwargs.get("batch_size", self.devo_config.eval_batch_size)),
            mode=str(kwargs.get("mode", "gradient")),
        )
        return attribution.numpy()

    def get_state(self) -> Mapping[str, Any]:
        return {
            "bundle_meta": None if self.bundle_meta is None else self.bundle_meta.to_dict(),
            "model_spec": dict(self._model_spec or {}),
            "training_history": list(self.training_history),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        bundle_meta = state.get("bundle_meta")
        if isinstance(bundle_meta, Mapping):
            self.bundle_meta = DatasetMeta.from_mapping(bundle_meta)
        else:
            self.bundle_meta = None

        model_spec = state.get("model_spec")
        self._model_spec = dict(model_spec) if isinstance(model_spec, Mapping) else None

        training_history = state.get("training_history", [])
        if isinstance(training_history, list):
            self.training_history = [dict(item) if isinstance(item, Mapping) else item for item in training_history]
        else:
            self.training_history = []

    def _save_additional_state(self, target: Path, payload: Mapping[str, Any]) -> None:
        del payload
        if self.model is None:
            return
        torch.save(self.model.state_dict(), _weights_path(target))

    def _load_additional_state(self, source: Path, payload: Mapping[str, Any]) -> None:
        del payload
        weights_path = _weights_path(source)
        if self._model_spec is None or not weights_path.exists():
            return

        self.model = self._build_model(
            window_length=int(self._model_spec["window_length"]),
            input_dim=int(self._model_spec["input_dim"]),
            output_dim=int(self._model_spec["output_dim"]),
            horizon=int(self._model_spec["horizon"]),
        )
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(device=self.device, dtype=self.dtype)
