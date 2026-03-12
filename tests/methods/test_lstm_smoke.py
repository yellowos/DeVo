"""Smoke tests for the methods-layer LSTM baseline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from methods.base import BaseMethod, KernelRecoveryNotSupportedError, create_method
from methods.baselines.lstm import LSTMMethod
from methods.utils.device import is_mps_available


def _make_bundle(*, target_mode: str) -> dict[str, object]:
    rng = np.random.default_rng(7)
    num_samples = 48
    window_length = 10
    input_dim = 3
    X = rng.normal(size=(num_samples, window_length, input_dim)).astype(np.float32)
    last_step = X[:, -1, :]
    pooled = X.mean(axis=1)

    if target_mode == "single":
        Y = (0.6 * last_step[:, 0] - 0.3 * pooled[:, 1] + 0.1).astype(np.float32).reshape(num_samples, 1, 1)
        output_dim = 1
        horizon = 1
    elif target_mode == "multi":
        y1 = 0.4 * last_step[:, 0] + 0.2 * pooled[:, 1]
        y2 = -0.3 * last_step[:, 2] + 0.1 * pooled[:, 0]
        Y = np.stack([y1, y2], axis=-1).astype(np.float32)
        output_dim = 2
        horizon = 1
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    return {
        "train": {"X": X[:32], "Y": Y[:32], "sample_id": np.arange(32)},
        "val": {"X": X[32:40], "Y": Y[32:40], "sample_id": np.arange(8)},
        "test": {"X": X[40:], "Y": Y[40:], "sample_id": np.arange(8)},
        "meta": {
            "dataset_name": f"toy_{target_mode}",
            "task_family": "nonlinear",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "window_length": window_length,
            "horizon": horizon,
            "split_protocol": "unit_test_protocol",
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
        },
        "artifacts": {},
    }


class LSTMBaselineSmokeTest(unittest.TestCase):
    def test_lstm_fit_predict_save_load_and_export(self) -> None:
        bundle = _make_bundle(target_mode="single")
        created = create_method("lstm")
        self.assertIsInstance(created, LSTMMethod)
        method = LSTMMethod(
            config={
                "hidden_size": 12,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "batch_size": 8,
                "max_epochs": 3,
                "learning_rate": 5e-3,
                "early_stopping_patience": 3,
            }
        )

        result = method.fit(bundle)
        predictions = method.predict(bundle["test"]["X"])

        self.assertTrue(method.is_fitted)
        self.assertEqual(predictions.shape, bundle["test"]["Y"].shape)
        self.assertFalse(method.supports_kernel_recovery())
        self.assertTrue(result.metadata["supports_input_gradients"])
        self.assertIn(method.runtime.device_type, {"mps", "cpu"})
        if is_mps_available():
            self.assertEqual(method.runtime.device_type, "mps")
        else:
            self.assertEqual(method.runtime.device_type, "cpu")

        with self.assertRaises(KernelRecoveryNotSupportedError):
            method.recover_kernels()

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = method.save(Path(tmp_dir) / "lstm_state.json")
            loaded = BaseMethod.load(save_path)
            reloaded_predictions = loaded.predict(bundle["test"]["X"])
            np.testing.assert_allclose(predictions, reloaded_predictions, atol=1e-6)

            artifacts = method.export_artifacts(Path(tmp_dir) / "artifacts")
            self.assertIn("state", artifacts)
            self.assertTrue((Path(tmp_dir) / "artifacts" / "lstm_method.json").exists())
            self.assertTrue((Path(tmp_dir) / "artifacts" / "model_weights.pt").exists())

    def test_lstm_multi_output_and_input_gradients(self) -> None:
        bundle = _make_bundle(target_mode="multi")
        method = LSTMMethod(
            config={
                "hidden_size": 10,
                "num_layers": 1,
                "dropout": 0.0,
                "batch_size": 8,
                "max_epochs": 2,
                "learning_rate": 1e-2,
                "early_stopping_patience": 2,
            }
        )
        method.fit(bundle)

        predictions = method.predict(bundle["test"]["X"][:3])
        self.assertEqual(predictions.shape, (3, 2))

        inputs = method.prepare_inputs(bundle["test"]["X"][:2], requires_grad=True)
        outputs = method.forward_tensor(inputs, reshape_output=False)
        scalar = outputs.sum()
        scalar.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertEqual(tuple(inputs.grad.shape), (2, 10, 3))
        self.assertTrue(np.isfinite(inputs.grad.detach().cpu().numpy()).all())
        self.assertGreater(float(inputs.grad.detach().abs().sum().cpu().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
