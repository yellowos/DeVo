"""Smoke tests for ARX / VAR baselines."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from methods.base import BaseMethod, create_method, get_method_class


def _build_linear_bundle() -> dict[str, object]:
    rng = np.random.default_rng(7)
    window_length = 3
    input_dim = 2
    horizon = 2
    output_dim = 2

    coefficient_tensor = np.array(
        [
            [
                [[0.8, -0.1], [0.2, 0.4], [-0.3, 0.6]],
                [[-0.5, 0.7], [0.1, -0.2], [0.3, 0.5]],
            ],
            [
                [[0.4, 0.2], [-0.6, 0.3], [0.9, -0.1]],
                [[0.2, -0.4], [0.5, 0.8], [-0.7, 0.6]],
            ],
        ],
        dtype=np.float64,
    )
    intercept = np.array([[0.1, -0.2], [0.05, 0.3]], dtype=np.float64)

    def _make_split(num_samples: int) -> dict[str, np.ndarray]:
        x = rng.normal(size=(num_samples, window_length, input_dim))
        y = np.einsum("nli,holi->nho", x, coefficient_tensor) + intercept[None, :, :]
        return {"X": x, "Y": y, "sample_id": np.arange(num_samples)}

    return {
        "train": _make_split(96),
        "val": _make_split(24),
        "test": _make_split(24),
        "meta": {
            "dataset_name": "toy_linear_bundle",
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


class ARXVARSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = _build_linear_bundle()

    def _run_roundtrip(self, method_name: str) -> None:
        method = create_method(method_name, alpha=1e-8, fit_intercept=True)
        result = method.fit(self.bundle)

        self.assertTrue(method.is_fitted)
        self.assertEqual(result.training_summary["model_family"], method_name)
        self.assertEqual(result.training_summary["coefficient_matrix_shape"], [4, 6])

        prediction = method.predict(self.bundle["test"]["X"])
        self.assertEqual(prediction.shape, self.bundle["test"]["Y"].shape)
        np.testing.assert_allclose(prediction, self.bundle["test"]["Y"], atol=1e-5, rtol=1e-5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            state_path = method.save(tmp_path / f"{method_name}_state.json")
            loaded = BaseMethod.load(state_path)
            loaded_prediction = loaded.predict(self.bundle["test"]["X"])
            np.testing.assert_allclose(loaded_prediction, prediction, atol=1e-8, rtol=1e-8)

            artifacts = method.export_artifacts(tmp_path / "artifacts")
            coefficient_matrix_path = Path(artifacts["coefficient_matrix"].path)
            coefficient_tensor_path = Path(artifacts["coefficient_tensor"].path)
            metadata_path = Path(artifacts["metadata"].path)

            self.assertTrue(coefficient_matrix_path.exists())
            self.assertTrue(coefficient_tensor_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(np.load(coefficient_matrix_path).shape, (4, 6))
            self.assertEqual(np.load(coefficient_tensor_path).shape, (2, 2, 3, 2))

            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["kernel_recovery_semantics"], result.training_summary["kernel_recovery_semantics"])
            self.assertEqual(metadata["equation"], "y_flat = x_flat @ coefficient_matrix.T + intercept_vector")

    def test_registry_can_lazy_load_arx_and_var(self) -> None:
        self.assertEqual(get_method_class("arx").METHOD_NAME, "arx")
        self.assertEqual(get_method_class("var").METHOD_NAME, "var")

    def test_arx_smoke(self) -> None:
        self._run_roundtrip("arx")

    def test_var_smoke(self) -> None:
        self._run_roundtrip("var")


if __name__ == "__main__":
    unittest.main()
