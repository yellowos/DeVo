"""Smoke tests for the Laguerre-Volterra baseline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from methods import BaseMethod
from methods.baselines.laguerre_volterra import LaguerreVolterraMethod, build_discrete_laguerre_basis


def _synthetic_bundle() -> tuple[dict[str, object], dict[str, float | int]]:
    config = {
        "laguerre_pole": 0.55,
        "num_laguerre": 3,
        "volterra_order": 2,
        "ridge_lambda": 1e-10,
    }
    num_samples = 72
    window_length = 16
    rng = np.random.RandomState(7)
    X = rng.randn(num_samples, window_length, 1)
    basis = build_discrete_laguerre_basis(window_length, config["num_laguerre"], config["laguerre_pole"])
    lagged = X[:, ::-1, :]
    states = np.einsum("nlc,kl->nck", lagged, basis, optimize=True)
    z = states.reshape(num_samples, -1)

    y = 0.45 + 1.10 * z[:, 0] - 0.35 * z[:, 1] + 0.60 * z[:, 0] * z[:, 1] + 0.25 * z[:, 2] ** 2
    Y = y[:, np.newaxis, np.newaxis]
    bundle = {
        "train": {
            "X": X[:48],
            "Y": Y[:48],
            "sample_id": np.arange(48),
        },
        "val": {
            "X": X[48:60],
            "Y": Y[48:60],
            "sample_id": np.arange(12),
        },
        "test": {
            "X": X[60:],
            "Y": Y[60:],
            "sample_id": np.arange(12),
        },
        "meta": {
            "dataset_name": "laguerre_volterra_toy",
            "task_family": "nonlinear",
            "input_dim": 1,
            "output_dim": 1,
            "window_length": window_length,
            "horizon": 1,
            "split_protocol": "unit_test_protocol",
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
        },
        "artifacts": {},
    }
    return bundle, config


class LaguerreVolterraSmokeTest(unittest.TestCase):
    def test_fit_predict_recover_save_load_and_export(self) -> None:
        bundle, config = _synthetic_bundle()
        method = LaguerreVolterraMethod(config=config)

        result = method.fit(bundle)
        self.assertTrue(method.is_fitted)
        self.assertEqual(result.training_summary["feature_count"], 10)

        test_X = np.asarray(bundle["test"]["X"])
        test_Y = np.asarray(bundle["test"]["Y"])
        predictions = method.predict(test_X)
        self.assertEqual(predictions.shape, test_Y.shape)
        self.assertLess(float(np.mean((predictions - test_Y) ** 2)), 1e-12)

        recovery = method.recover_kernels(map_to_time_domain=True)
        self.assertIn("representation", recovery.kernels)
        self.assertEqual(recovery.kernels["representation"]["type"], "laguerre_volterra_basis_coefficients")
        self.assertEqual(recovery.kernels["orders"]["1"]["coefficients"].shape, (1, 1, 1, 3))
        self.assertEqual(recovery.kernels["orders"]["2"]["coefficients"].shape, (1, 1, 1, 3, 1, 3))
        self.assertIn("order_1", recovery.kernels["time_domain_kernels"])
        self.assertIn("order_2", recovery.kernels["time_domain_kernels"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = method.save(Path(tmp_dir) / "laguerre_volterra.json")
            loaded = BaseMethod.load(state_path)
            self.assertIsInstance(loaded, LaguerreVolterraMethod)
            reloaded_predictions = loaded.predict(test_X)
            self.assertTrue(np.allclose(predictions, reloaded_predictions))

            artifacts = loaded.export_artifacts(Path(tmp_dir) / "artifacts")
            self.assertTrue(Path(artifacts["summary"].path).exists())
            self.assertTrue(Path(artifacts["basis"].path).exists())
            self.assertTrue(Path(artifacts["coefficients"].path).exists())
            self.assertTrue(Path(artifacts["feature_structure"].path).exists())


if __name__ == "__main__":
    unittest.main()
