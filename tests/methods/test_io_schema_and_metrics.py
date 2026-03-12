from __future__ import annotations

import unittest

import numpy as np

from data.adapters.base import DataProtocolError
from methods.base import load_dataset_bundle
from methods.metrics import compute_prediction_metrics


def _bundle_payload() -> dict[str, object]:
    def _split(sample_ids: list[int], *, run_id: str) -> dict[str, object]:
        count = len(sample_ids)
        starts = np.arange(count, dtype=np.int64)
        return {
            "X": np.ones((count, 3, 1), dtype=np.float32),
            "Y": np.ones((count, 1, 1), dtype=np.float32),
            "sample_id": np.asarray(sample_ids),
            "run_id": np.asarray([run_id] * count, dtype=object),
            "timestamp": np.arange(count, dtype=np.float64),
            "window_start": starts,
            "window_end": starts + 2,
            "target_index": starts + 3,
        }

    return {
        "train": _split([0, 1, 2], run_id="train_run"),
        "val": _split([3, 4], run_id="val_run"),
        "test": _split([5, 6], run_id="test_run"),
        "meta": {
            "dataset_name": "toy_schema",
            "task_family": "nonlinear",
            "input_dim": 1,
            "output_dim": 1,
            "window_length": 3,
            "horizon": 1,
            "split_protocol": "unit_test_protocol",
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
        },
        "artifacts": {},
    }


class IOSchemaAndMetricsTest(unittest.TestCase):
    def test_schema_rejects_overlapping_splits(self) -> None:
        payload = _bundle_payload()
        payload["val"] = {
            **payload["val"],
            "run_id": np.asarray(["train_run", "train_run"], dtype=object),
            "window_start": np.asarray([1, 10]),
            "window_end": np.asarray([3, 12]),
            "target_index": np.asarray([4, 13]),
        }
        with self.assertRaises(DataProtocolError):
            load_dataset_bundle(payload)

    def test_schema_rejects_cross_run_windows(self) -> None:
        payload = _bundle_payload()
        payload["train"] = {
            **payload["train"],
            "window_start": np.asarray([0, 1, 2]),
            "window_end": np.asarray([2, 3, 4]),
            "target_index": np.asarray([3, 4, 5]),
            "target_run_id": np.asarray(["train_run", "other_run", "train_run"], dtype=object),
        }
        with self.assertRaises(DataProtocolError):
            load_dataset_bundle(payload)

    def test_metric_consistency_train_eval_collect(self) -> None:
        from experiments.nonlinear.exp01_prediction_benchmark.collect import _aggregate
        from experiments.nonlinear.exp01_prediction_benchmark.run import _compute_metrics

        y_true = np.array([[[1.0]], [[2.0]]], dtype=np.float64)
        y_pred = np.array([[[1.5]], [[2.5]]], dtype=np.float64)
        facade_metrics = compute_prediction_metrics(y_true, y_pred, domain="native")
        run_metrics = _compute_metrics(y_true, y_pred)
        rows = _aggregate(
            [
                {
                    "dataset": "toy",
                    "method": "devo",
                    "status": "success",
                    "metrics": {
                        "nmse": facade_metrics["nmse"],
                        "rmse": facade_metrics["rmse"],
                    },
                }
            ],
            ["toy"],
            ["devo"],
        )
        self.assertAlmostEqual(facade_metrics["mse"], run_metrics["mse"])
        self.assertAlmostEqual(facade_metrics["rmse"], run_metrics["rmse"])
        self.assertAlmostEqual(facade_metrics["nmse"], run_metrics["nmse"])
        self.assertAlmostEqual(rows[0]["nmse_mean"], facade_metrics["nmse"])
        self.assertAlmostEqual(rows[0]["rmse_mean"], facade_metrics["rmse"])


if __name__ == "__main__":
    unittest.main()
