"""Smoke tests for the methods-layer MLP baseline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from methods.base import KernelRecoveryNotSupportedError, get_method_class


class MLPMethodSmokeTest(unittest.TestCase):
    def _make_bundle(self) -> dict[str, object]:
        rng = np.random.default_rng(7)
        window_length = 8
        input_dim = 2
        horizon = 2
        output_dim = 1

        def _split(num_samples: int) -> dict[str, np.ndarray]:
            X = rng.normal(size=(num_samples, window_length, input_dim)).astype(np.float32)
            Y = np.stack(
                [
                    0.6 * X[:, -1, 0] - 0.2 * X[:, -2, 1],
                    0.4 * X[:, -3, 0] + 0.1 * X[:, -1, 1],
                ],
                axis=1,
            ).astype(np.float32)
            return {
                "X": X,
                "Y": Y.reshape(num_samples, horizon, output_dim),
                "sample_id": np.arange(num_samples),
            }

        return {
            "train": _split(64),
            "val": _split(16),
            "test": _split(16),
            "meta": {
                "dataset_name": "mlp_smoke",
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

    def test_mlp_can_fit_predict_save_load_and_export(self):
        method_cls = get_method_class("mlp")
        method = method_cls(
            config={
                "hidden_dim": 32,
                "num_hidden_layers": 2,
                "dropout": 0.0,
                "epochs": 8,
                "batch_size": 16,
                "learning_rate": 5e-3,
                "log_every": 4,
                "verbose": False,
            }
        )
        bundle = self._make_bundle()

        result = method.fit(bundle)
        self.assertTrue(method.is_fitted)
        self.assertEqual(result.training_summary["loss_family"], "mse")
        self.assertFalse(method.supports_kernel_recovery())

        predictions = method.predict(bundle["test"]["X"])
        self.assertEqual(predictions.shape, bundle["test"]["Y"].shape)
        self.assertEqual(predictions.dtype, np.float32)

        with self.assertRaises(KernelRecoveryNotSupportedError):
            method.recover_kernels()

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = method.save(Path(tmp_dir) / "mlp_checkpoint.pt")
            loaded = method_cls.load(checkpoint_path)
            reloaded_predictions = loaded.predict(bundle["test"]["X"])
            np.testing.assert_allclose(predictions, reloaded_predictions, rtol=1e-5, atol=1e-5)

            artifacts = loaded.export_artifacts(Path(tmp_dir) / "artifacts")
            self.assertIn("config", artifacts)
            self.assertIn("training_summary", artifacts)
            self.assertIn("metadata", artifacts)
            self.assertTrue(Path(artifacts["config"]).exists())
            self.assertTrue(Path(artifacts["training_summary"]).exists())
            self.assertTrue(Path(artifacts["metadata"]).exists())


if __name__ == "__main__":
    unittest.main()
