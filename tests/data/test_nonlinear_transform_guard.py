from __future__ import annotations

import unittest

import numpy as np

from data.adapters import NonlinearAdapter
from data.builders.nonlinear_builder import apply_train_only_standardization


class NonlinearTransformGuardTest(unittest.TestCase):
    def test_scaler_fit_only_on_train_split(self) -> None:
        train = {
            "X": np.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]], dtype=np.float64),
            "Y": np.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]], dtype=np.float64),
            "sample_id": np.arange(4),
        }
        val = {
            "X": np.array([[[10.0]], [[11.0]]], dtype=np.float64),
            "Y": np.array([[[10.0]], [[11.0]]], dtype=np.float64),
            "sample_id": np.arange(2),
        }
        test = {
            "X": np.array([[[20.0]], [[21.0]]], dtype=np.float64),
            "Y": np.array([[[20.0]], [[21.0]]], dtype=np.float64),
            "sample_id": np.arange(2),
        }

        train_scaled, val_scaled, test_scaled, stats = apply_train_only_standardization(
            train=train,
            val=val,
            test=test,
        )

        self.assertAlmostEqual(float(stats.mean.reshape(-1)[0]), 1.5)
        self.assertAlmostEqual(float(train_scaled["X"].mean()), 0.0, places=6)
        self.assertGreater(float(test_scaled["X"].mean()), 10.0)
        self.assertAlmostEqual(float(np.asarray(train["X"]).mean()), 1.5)

        bundle = NonlinearAdapter.build_bundle(
            dataset_name="duffing",
            train=train_scaled,
            val=val_scaled,
            test=test_scaled,
            meta={},
            artifacts={},
        )
        provenance = bundle.meta.extras["transform_provenance"]
        self.assertEqual(provenance["source_split"], "train")
        self.assertEqual(bundle.artifacts.extra["transform_provenance"]["source_split"], "train")


if __name__ == "__main__":
    unittest.main()
