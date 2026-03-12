"""Smoke tests for methods-layer base protocol and utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from methods.base import BaseMethod, MethodResult, create_method, get_method_class, load_dataset_bundle, register_method
from methods.utils.device import is_mps_available, select_device


class DummyMethod(BaseMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bias = 0.0

    def fit(self, dataset_bundle, **kwargs):
        del kwargs
        bundle = self.normalize_dataset_bundle(dataset_bundle)
        train_y = np.asarray(bundle.train.Y, dtype=np.float32)
        self.bias = float(train_y.mean())
        self.training_summary = {
            "dataset_name": bundle.meta.dataset_name,
            "train_samples": bundle.train.num_samples,
        }
        self.is_fitted = True
        return MethodResult(
            predictions=None,
            model_state_path=self.model_state_path,
            training_summary=dict(self.training_summary),
            artifacts={},
        )

    def predict(self, X, **kwargs):
        del kwargs
        x = np.asarray(X)
        return np.full((len(x), 1), self.bias, dtype=np.float32)

    def export_artifacts(self, output_dir):
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        return {"export_dir": str(target)}

    def get_state(self):
        return {"bias": self.bias}

    def set_state(self, state):
        self.bias = float(state.get("bias", 0.0))


class MethodsProtocolSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_method("dummy_method_smoke", DummyMethod, overwrite=True)

    def test_base_method_can_fit_save_and_load(self):
        method = DummyMethod()
        bundle = {
            "train": {
                "X": np.ones((4, 8, 1), dtype=np.float32),
                "Y": np.full((4, 1), 2.0, dtype=np.float32),
                "sample_id": np.arange(4),
            },
            "val": {
                "X": np.ones((2, 8, 1), dtype=np.float32),
                "Y": np.full((2, 1), 2.0, dtype=np.float32),
                "sample_id": np.arange(2),
            },
            "test": {
                "X": np.ones((2, 8, 1), dtype=np.float32),
                "Y": np.full((2, 1), 2.0, dtype=np.float32),
                "sample_id": np.arange(2),
            },
            "meta": {
                "dataset_name": "toy_bundle",
                "task_family": "nonlinear",
                "input_dim": 1,
                "output_dim": 1,
                "window_length": 8,
                "horizon": 1,
                "split_protocol": "unit_test_protocol",
                "has_ground_truth_kernel": False,
                "has_ground_truth_gfrf": False,
            },
            "artifacts": {},
        }

        result = method.fit(bundle)
        self.assertTrue(method.is_fitted)
        self.assertIsInstance(result, MethodResult)
        self.assertEqual(result.training_summary["train_samples"], 4)

        predictions = method.predict(np.zeros((3, 8, 1), dtype=np.float32))
        self.assertEqual(predictions.shape, (3, 1))
        self.assertAlmostEqual(float(predictions[0, 0]), 2.0, places=5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = method.save(Path(tmp_dir) / "dummy_method.json")
            loaded = DummyMethod.load(state_path)
            self.assertAlmostEqual(loaded.bias, 2.0, places=5)
            self.assertTrue(loaded.is_fitted)

    def test_registry_can_register_lookup_and_create(self):
        self.assertIs(get_method_class("dummy_method_smoke"), DummyMethod)
        instance = create_method("dummy-method-smoke")
        self.assertIsInstance(instance, DummyMethod)

    def test_dataset_bundle_loader_reads_processed_manifest(self):
        processed_root = Path(__file__).resolve().parents[2] / "data" / "processed" / "nonlinear" / "duffing"
        bundle = load_dataset_bundle(processed_root)
        self.assertEqual(bundle.meta.dataset_name, "duffing")
        self.assertGreater(bundle.train.num_samples, 0)
        self.assertEqual(bundle.train.X.shape[-1], bundle.meta.input_dim)

    def test_device_policy_prefers_mps_then_cpu(self):
        context = select_device()
        self.assertIn(context.device_type, {"mps", "cpu"})
        if is_mps_available():
            self.assertEqual(context.device_type, "mps")
        else:
            self.assertEqual(context.device_type, "cpu")
        self.assertEqual(context.dtype_name, "float32")

        forced_mps = select_device(preferred_device="mps", preferred_dtype="float64")
        if is_mps_available():
            self.assertEqual(forced_mps.device_type, "mps")
            self.assertEqual(forced_mps.dtype_name, "float32")
        else:
            self.assertEqual(forced_mps.device_type, "cpu")


if __name__ == "__main__":
    unittest.main()
