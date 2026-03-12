"""Smoke tests for the methods-layer TCN baseline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from methods.base import BaseMethod, KernelRecoveryNotSupportedError, create_method, get_method_class
from methods.utils.device import is_mps_available

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is absent.
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required for TCN smoke tests.")
class TCNMethodSmokeTest(unittest.TestCase):
    def _make_split(self, rng: np.random.Generator, num_samples: int) -> dict[str, np.ndarray]:
        X = rng.normal(size=(num_samples, 16, 3)).astype(np.float32)
        Y = (
            0.60 * X[:, -1, 0]
            - 0.25 * X[:, -2, 1]
            + 0.15 * X[:, -4:, 2].mean(axis=1)
        ).astype(np.float32).reshape(num_samples, 1)
        return {
            "X": X,
            "Y": Y,
            "sample_id": np.arange(num_samples),
        }

    def _make_bundle(self) -> dict[str, object]:
        rng = np.random.default_rng(7)
        return {
            "train": self._make_split(rng, 24),
            "val": self._make_split(rng, 8),
            "test": self._make_split(rng, 8),
            "meta": {
                "dataset_name": "toy_tcn_bundle",
                "task_family": "hydraulic",
                "input_dim": 3,
                "output_dim": 1,
                "window_length": 16,
                "horizon": 1,
                "split_protocol": "unit_test_protocol",
                "has_ground_truth_kernel": False,
                "has_ground_truth_gfrf": False,
            },
            "artifacts": {},
        }

    def test_tcn_registered_and_uses_expected_device_policy(self):
        tcn_method = create_method("tcn")
        self.assertEqual(get_method_class("tcn").__name__, "TCNMethod")
        if is_mps_available():
            self.assertEqual(tcn_method.runtime.device_type, "mps")
        else:
            self.assertEqual(tcn_method.runtime.device_type, "cpu")
        self.assertEqual(tcn_method.runtime.dtype_name, "float32")

    def test_tcn_fit_predict_save_load_and_gradients(self):
        bundle = self._make_bundle()
        method = create_method(
            "tcn",
            config={
                "num_channels": [8, 8],
                "kernel_size": 3,
                "dilation_schedule": [1, 2],
                "dropout": 0.0,
                "epochs": 4,
                "batch_size": 6,
                "learning_rate": 5e-3,
            },
        )

        result = method.fit(bundle)
        self.assertTrue(method.is_fitted)
        self.assertFalse(method.supports_kernel_recovery())
        with self.assertRaises(KernelRecoveryNotSupportedError):
            method.recover_kernels()
        self.assertTrue(result.metadata["supports_local_input_gradients"])

        predictions = method.predict(bundle["test"]["X"])
        self.assertEqual(predictions.shape, bundle["test"]["Y"].shape)
        self.assertTrue(np.isfinite(predictions).all())

        tensor_predictions = method.predict_tensor(torch.tensor(bundle["test"]["X"][:2], dtype=torch.float32))
        self.assertEqual(tuple(tensor_predictions.shape), (2, 1))

        gradients = method.compute_input_gradients(bundle["test"]["X"][:3], target_index=0, batch_size=2)
        self.assertEqual(gradients.shape, (3, 16, 3))
        self.assertEqual(gradients.dtype, np.float32)
        self.assertTrue(np.isfinite(gradients).all())

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = method.save(Path(tmp_dir) / "tcn_state")
            loaded = BaseMethod.load(state_path)
            reloaded_predictions = loaded.predict(bundle["test"]["X"])
            np.testing.assert_allclose(predictions, reloaded_predictions, atol=1e-5, rtol=1e-5)

            artifacts = loaded.export_artifacts(Path(tmp_dir) / "artifacts")
            self.assertIn("training_summary", artifacts)
            self.assertTrue((Path(tmp_dir) / "artifacts" / "tcn_capabilities.json").exists())
