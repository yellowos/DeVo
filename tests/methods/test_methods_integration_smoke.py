"""Integration smoke tests across data-method boundaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.adapters.base import DataProtocolError
from methods.base import (
    BaseMethod,
    KernelRecoveryNotSupportedError,
    KernelRecoveryResult,
    create_method,
    load_dataset_bundle,
)
from methods.devo import DeVoConfig, DeVoMethod
from methods.baselines.laguerre_volterra import LaguerreVolterraMethod
from methods.utils import slice_dataset_bundle


def _build_bundle(*, seed: int = 7) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    window_length = 8
    input_dim = 2
    output_dim = 1
    horizon = 1

    def _split(num_samples: int) -> dict[str, np.ndarray]:
        x = rng.normal(size=(num_samples, window_length, input_dim)).astype(np.float32)
        y = (
            0.6 * x[:, -1, 0]
            - 0.2 * x[:, -1, 1]
            + 0.1 * x[:, 0, 0] * x[:, 1, 1]
        ).astype(np.float32)
        return {
            "X": x,
            "Y": y[:, None, None],
            "sample_id": np.arange(num_samples),
        }

    return {
        "train": _split(64),
        "val": _split(16),
        "test": _split(16),
        "meta": {
            "dataset_name": "toy_methods_bundle",
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


class MethodsIntegrationSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = _build_bundle()

    def test_methods_fit_predict_recover_and_roundtrip_on_shared_bundle(self) -> None:
        methods = [
            (
                "mlp",
                create_method(
                    "mlp",
                    config={
                        "hidden_dim": 16,
                        "num_hidden_layers": 1,
                        "dropout": 0.0,
                        "epochs": 3,
                        "batch_size": 8,
                        "learning_rate": 4e-3,
                        "verbose": False,
                    },
                ),
            ),
            (
                "laguerre",
                LaguerreVolterraMethod(
                    config={
                        "laguerre_pole": 0.55,
                        "num_laguerre": 2,
                        "volterra_order": 2,
                        "ridge_lambda": 1e-10,
                    }
                ),
            ),
            (
                "devo",
                DeVoMethod(
                    config=DeVoConfig(
                        orders=(1, 2),
                        num_branches=1,
                        epochs=2,
                        batch_size=8,
                        eval_batch_size=8,
                        learning_rate=5e-3,
                        feature_chunk_size=128,
                        verbose=False,
                        log_every=1,
                    )
                ),
            ),
        ]

        for name, method in methods:
            with self.subTest(method=name):
                result = method.fit(self.bundle)
                self.assertTrue(method.is_fitted)
                self.assertEqual(result.training_summary["dataset_name"], self.bundle["meta"]["dataset_name"])

                pred = method.predict(self.bundle["test"]["X"])
                self.assertEqual(pred.shape, self.bundle["test"]["Y"].shape)

                if method.supports_kernel_recovery():
                    recovery = method.recover_kernels()
                    self.assertIsInstance(recovery, KernelRecoveryResult)
                    self.assertIn("orders", recovery.kernels)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = Path(tmp_dir) / f"{name}_state.json"
                    saved = method.save(path)
                    restored = method.__class__.load(saved)
                    reloaded_pred = restored.predict(self.bundle["test"]["X"][:4])
                    np.testing.assert_allclose(pred[:4], reloaded_pred, rtol=1e-5, atol=1e-5)

    def test_recovery_not_supported_is_explicit(self) -> None:
        method = create_method("mlp", config={"epochs": 1, "verbose": False})
        with self.assertRaises(KernelRecoveryNotSupportedError):
            method.recover_kernels()

    def test_missing_bundle_meta_fails_fast(self) -> None:
        broken = dict(self.bundle)
        del broken["meta"]
        method = create_method("mlp", config={"epochs": 1, "verbose": False})
        with self.assertRaises(DataProtocolError):
            method.fit(broken)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[2] / "data" / "processed" / "nonlinear" / "duffing" / "duffing_processed_manifest.json").exists(),
        "duffing processed manifest is not available.",
    )
    def test_real_bundle_smoke_with_methods(self) -> None:
        manifest = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "processed"
            / "nonlinear"
            / "duffing"
            / "duffing_processed_manifest.json"
        )
        full_bundle = load_dataset_bundle(manifest)
        mini_bundle = slice_dataset_bundle(full_bundle, train_size=40, val_size=8, test_size=8)
        method = create_method(
            "mlp",
            config={
                "hidden_dim": 8,
                "num_hidden_layers": 1,
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": 1e-2,
                "verbose": False,
            },
        )
        result = method.fit(mini_bundle)
        self.assertTrue(result.training_summary.get("train_samples", 0) > 0)
        pred = method.predict(mini_bundle.test.X[:4])
        self.assertEqual(pred.shape[0], 4)
        self.assertEqual(pred.shape[1], mini_bundle.meta.output_dim)


if __name__ == "__main__":
    unittest.main()
