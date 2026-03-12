"""Tests for code audit fixes: seed handling, validation fallback, and data checks."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from methods.utils.runtime import set_random_seed


class SetRandomSeedTest(unittest.TestCase):
    """Verify that set_random_seed covers all relevant RNGs."""

    def test_torch_manual_seed_is_set(self):
        set_random_seed(123)
        a = torch.randn(5)
        set_random_seed(123)
        b = torch.randn(5)
        torch.testing.assert_close(a, b)

    def test_numpy_seed_is_set(self):
        set_random_seed(456)
        a = np.random.rand(5)
        set_random_seed(456)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_cuda_seed_is_set_when_available(self):
        """If CUDA is available, manual_seed_all must be invoked without error."""
        # Should not raise regardless of CUDA availability.
        set_random_seed(789)


class MLPSeedHandlingTest(unittest.TestCase):
    """MLP baseline must expose and honour a seed config key."""

    def test_default_config_contains_seed(self):
        from methods.baselines.mlp.mlp_method import _DEFAULT_CONFIG

        self.assertIn("seed", _DEFAULT_CONFIG)
        self.assertIsInstance(_DEFAULT_CONFIG["seed"], int)

    def test_two_fits_with_same_seed_are_deterministic(self):
        from methods.base import get_method_class

        bundle = _make_tiny_bundle(rng_seed=10)
        cls = get_method_class("mlp")

        m1 = cls(config={"epochs": 3, "batch_size": 8, "verbose": False, "seed": 99})
        m1.fit(bundle)
        p1 = m1.predict(bundle["test"]["X"])

        m2 = cls(config={"epochs": 3, "batch_size": 8, "verbose": False, "seed": 99})
        m2.fit(bundle)
        p2 = m2.predict(bundle["test"]["X"])

        np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-5)


class TCNSeedHandlingTest(unittest.TestCase):
    """TCN baseline must expose and honour a seed config key."""

    def test_default_config_contains_seed(self):
        from methods.baselines.tcn.method import TCNMethod

        self.assertIn("seed", TCNMethod._DEFAULT_CONFIG)
        self.assertIsInstance(TCNMethod._DEFAULT_CONFIG["seed"], int)

    def test_two_fits_with_same_seed_are_deterministic(self):
        from methods.base import get_method_class

        bundle = _make_tiny_bundle(rng_seed=20)
        cls = get_method_class("tcn")

        m1 = cls(config={"epochs": 2, "batch_size": 8, "seed": 77})
        m1.fit(bundle)
        p1 = m1.predict(bundle["test"]["X"])

        m2 = cls(config={"epochs": 2, "batch_size": 8, "seed": 77})
        m2.fit(bundle)
        p2 = m2.predict(bundle["test"]["X"])

        np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-5)


class CPVolterraValidationFallbackTest(unittest.TestCase):
    """CP-Volterra must reject empty validation splits."""

    def test_empty_val_raises(self):
        from methods.base import get_method_class

        bundle = _make_tiny_bundle(rng_seed=30, val_samples=0)
        cls = get_method_class("cp_volterra")
        method = cls(config={"max_order": 2, "rank": 2, "epochs": 1})
        with self.assertRaises(ValueError):
            method.fit(bundle)


class TTVolterraValidationFallbackTest(unittest.TestCase):
    """TT-Volterra must reject empty validation splits."""

    def test_empty_val_raises(self):
        from methods.base import get_method_class

        bundle = _make_tiny_bundle(rng_seed=40, val_samples=0)
        cls = get_method_class("tt_volterra")
        method = cls(config={"orders": [1, 2], "epochs": 1, "batch_size": 4})
        with self.assertRaises(ValueError):
            method.fit(bundle)


class DuplicateIndexCheckTest(unittest.TestCase):
    """Within-split duplicate indices must be flagged as errors, not warnings."""

    def test_duplicate_indices_produce_error(self):
        from data.checks.unified_data_layer_checks import CheckReport, _check_split_disjointness

        report = CheckReport()
        split_payloads = {
            "train": {"X": np.zeros((4, 2)), "Y": np.zeros((4, 1))},
            "val": {"X": np.zeros((2, 2)), "Y": np.zeros((2, 1))},
            "test": {"X": np.zeros((2, 2)), "Y": np.zeros((2, 1))},
        }
        # train indices contain a duplicate (0 appears twice)
        split_indices = {"split_indices": {"train": [0, 1, 1, 2], "val": [3, 4], "test": [5, 6]}}
        _check_split_disjointness(report, split_payloads, split_indices, "test_ds")
        # The duplicate must be reported as an error, not just a warning.
        error_texts = [e for e in report.errors if "duplicate" in e.lower()]
        self.assertTrue(len(error_texts) > 0, "Duplicate indices should generate an error")


class IoSchemaPickleTest(unittest.TestCase):
    """Numerical arrays must load without allow_pickle."""

    def test_pickle_allowed_fields_does_not_include_X_or_Y(self):
        from methods.base.io_schema import _PICKLE_ALLOWED_FIELDS

        self.assertNotIn("X", _PICKLE_ALLOWED_FIELDS)
        self.assertNotIn("Y", _PICKLE_ALLOWED_FIELDS)
        self.assertIn("sample_id", _PICKLE_ALLOWED_FIELDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_bundle(
    *,
    rng_seed: int = 0,
    train_samples: int = 32,
    val_samples: int = 8,
    test_samples: int = 8,
) -> dict:
    rng = np.random.default_rng(rng_seed)
    window_length = 4
    input_dim = 2
    horizon = 1
    output_dim = 1

    def _split(n: int) -> dict:
        X = rng.normal(size=(n, window_length, input_dim)).astype(np.float32)
        Y = (0.5 * X[:, -1:, :1]).astype(np.float32)
        return {"X": X, "Y": Y, "sample_id": np.arange(n)}

    return {
        "train": _split(train_samples),
        "val": _split(val_samples),
        "test": _split(test_samples),
        "meta": {
            "dataset_name": "audit_test",
            "task_family": "nonlinear",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "window_length": window_length,
            "horizon": horizon,
            "split_protocol": "unit_test",
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
        },
        "artifacts": {},
    }


if __name__ == "__main__":
    unittest.main()
