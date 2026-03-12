"""Minimal smoke coverage for the NARMAX baseline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from methods.base import create_method, load_dataset_bundle
from methods.baselines.narmax import NARMAXMethod


def _slice_optional(value, stop: int):
    if value is None:
        return None
    return value[:stop]


def _subset_bundle(bundle, *, train_size: int, val_size: int, test_size: int) -> dict[str, object]:
    return {
        "train": {
            "X": bundle.train.X[:train_size],
            "Y": bundle.train.Y[:train_size],
            "sample_id": _slice_optional(bundle.train.sample_id, train_size),
            "run_id": _slice_optional(bundle.train.run_id, train_size),
            "timestamp": _slice_optional(bundle.train.timestamp, train_size),
            "meta": dict(bundle.train.meta),
        },
        "val": {
            "X": bundle.val.X[:val_size],
            "Y": bundle.val.Y[:val_size],
            "sample_id": _slice_optional(bundle.val.sample_id, val_size),
            "run_id": _slice_optional(bundle.val.run_id, val_size),
            "timestamp": _slice_optional(bundle.val.timestamp, val_size),
            "meta": dict(bundle.val.meta),
        },
        "test": {
            "X": bundle.test.X[:test_size],
            "Y": bundle.test.Y[:test_size],
            "sample_id": _slice_optional(bundle.test.sample_id, test_size),
            "run_id": _slice_optional(bundle.test.run_id, test_size),
            "timestamp": _slice_optional(bundle.test.timestamp, test_size),
            "meta": dict(bundle.test.meta),
        },
        "meta": bundle.meta.to_dict(),
        "artifacts": bundle.artifacts.to_dict(),
    }


class NARMAXSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        processed_root = Path(__file__).resolve().parents[3] / "data" / "processed" / "nonlinear" / "duffing"
        full_bundle = load_dataset_bundle(processed_root)
        cls.bundle = _subset_bundle(full_bundle, train_size=256, val_size=64, test_size=64)
        cls.config = {
            "input_lags": 6,
            "output_lags": 0,
            "moving_average_lags": 0,
            "polynomial_order": 2,
            "max_base_terms": 6,
            "max_terms": 24,
            "ranking_sample_size": 128,
            "ridge_alpha": 1e-5,
        }

    def test_registry_lazy_loads_narmax(self) -> None:
        method = create_method("narmax", config=self.config)
        self.assertIsInstance(method, NARMAXMethod)

    def test_fit_predict_export_recover_and_load(self) -> None:
        method = NARMAXMethod(config=self.config)
        result = method.fit(self.bundle)
        self.assertTrue(method.is_fitted)
        self.assertEqual(result.training_summary["dataset_name"], "duffing")

        test_x = np.asarray(self.bundle["test"]["X"])
        predictions = method.predict(test_x)
        self.assertEqual(predictions.shape, np.asarray(self.bundle["test"]["Y"]).shape)
        self.assertTrue(np.isfinite(predictions).all())

        recovery = method.recover_kernels()
        self.assertTrue(method.supports_kernel_recovery())
        self.assertEqual(recovery.kernels["representation_type"], "narmax_structural_coefficients")
        self.assertIn("input_polynomial_terms", recovery.kernels)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "exports"
            artifacts = method.export_artifacts(output_dir)
            for artifact_path in artifacts.values():
                self.assertTrue(Path(artifact_path).exists())

            state_path = method.save(Path(tmp_dir) / "narmax_state.json")
            reloaded = NARMAXMethod.load(state_path)
            restored_predictions = reloaded.predict(test_x)
            self.assertTrue(np.allclose(predictions, restored_predictions))


if __name__ == "__main__":
    unittest.main()
