from __future__ import annotations

import unittest

import numpy as np

from methods.devo import DeVoConfig, DeVoMethod
from methods.devo.canonical_features import build_aligned_windows, infer_alignment_from_windowed_batch, validate_alignment
from methods.devo.trainer import DeVoTrainingError


def _make_bundle(*, with_nan: bool = False) -> dict[str, object]:
    x_train = np.arange(24, dtype=np.float32).reshape(8, 3, 1)
    y_train = np.arange(8, dtype=np.float32).reshape(8, 1, 1)
    if with_nan:
        x_train = x_train.copy()
        x_train[0, 0, 0] = np.nan

    def _split(x: np.ndarray, y: np.ndarray, *, run_id: str, offset: int) -> dict[str, object]:
        sample_count = x.shape[0]
        starts = np.arange(offset, offset + sample_count, dtype=np.int64)
        return {
            "X": x,
            "Y": y,
            "sample_id": np.arange(sample_count),
            "run_id": np.asarray([run_id] * sample_count, dtype=object),
            "window_start": starts,
            "window_end": starts + 2,
            "target_index": starts + 3,
        }

    return {
        "train": _split(x_train, y_train, run_id="run_train", offset=0),
        "val": _split(
            np.arange(12, dtype=np.float32).reshape(4, 3, 1),
            np.arange(4, dtype=np.float32).reshape(4, 1, 1),
            run_id="run_val",
            offset=100,
        ),
        "test": _split(
            np.arange(12, dtype=np.float32).reshape(4, 3, 1),
            np.arange(4, dtype=np.float32).reshape(4, 1, 1),
            run_id="run_test",
            offset=200,
        ),
        "meta": {
            "dataset_name": "toy_devo",
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


class DeVoAlignmentAndContextTest(unittest.TestCase):
    def test_partial_alignment_infers_window_start_from_window_end(self) -> None:
        x = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
        y = np.arange(4, dtype=np.float32).reshape(2, 2, 1)
        alignment = {"window_end": np.asarray([2, 5], dtype=np.int64)}

        payload = infer_alignment_from_windowed_batch(
            num_samples=2,
            window_length=3,
            horizon=2,
            alignment=alignment,
        )

        np.testing.assert_array_equal(payload["window_start"], np.asarray([0, 3], dtype=np.int64))
        np.testing.assert_array_equal(payload["target_index"], np.asarray([4, 7], dtype=np.int64))
        validate_alignment(
            x,
            y,
            window_length=3,
            input_dim=1,
            horizon=2,
            output_dim=1,
            alignment=alignment,
        )

    def test_window_target_alignment_h1(self) -> None:
        payload = build_aligned_windows(np.arange(8, dtype=np.float64), window_length=3, horizon=1)
        np.testing.assert_array_equal(payload["X"][0].reshape(-1), np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(payload["Y"][0].reshape(-1), np.array([3.0]))
        self.assertEqual(int(payload["window_start"][0]), 0)
        self.assertEqual(int(payload["window_end"][0]), 2)
        self.assertEqual(int(payload["target_index"][0]), 3)

    def test_window_target_alignment_h5(self) -> None:
        payload = build_aligned_windows(np.arange(10, dtype=np.float64), window_length=3, horizon=5)
        np.testing.assert_array_equal(payload["X"][0].reshape(-1), np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(payload["Y"][0].reshape(-1), np.array([3.0, 4.0, 5.0, 6.0, 7.0]))
        self.assertEqual(int(payload["target_index"][0]), 7)
        self.assertEqual(int(payload["horizon"][0]), 5)

    def test_trainer_reports_nan_with_context(self) -> None:
        method = DeVoMethod(
            config=DeVoConfig(
                orders=(1,),
                num_branches=1,
                epochs=1,
                batch_size=2,
                eval_batch_size=2,
                learning_rate=1e-3,
                feature_chunk_size=32,
                verbose=False,
            )
        )
        with self.assertRaises(DeVoTrainingError) as captured:
            method.fit(_make_bundle(with_nan=True), run_context={"run_id": "nan_run"})
        message = str(captured.exception)
        self.assertIn("nan_run", message)
        self.assertIn("epoch", message)
        self.assertIn("batch_index", message)


if __name__ == "__main__":
    unittest.main()
