from __future__ import annotations

import unittest

import numpy as np

from experiments.tep.exp01_five_unit_fault_isolation.run import (
    aggregate_unit_scores,
    build_windows,
    canonical_expected_units,
    count_selected_evaluable_runs,
    compute_run_metrics,
    select_fault_runs,
    truth_row_is_evaluable,
)


class TEPExp01FiveUnitIsolationTest(unittest.TestCase):
    def test_build_windows_handles_variable_length_and_horizon(self) -> None:
        run_values = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
        time_index = np.arange(12, dtype=np.int32)

        windows = build_windows(
            run_values=run_values,
            time_index=time_index,
            window_length=4,
            horizon=3,
        )

        self.assertEqual(windows["X"].shape, (6, 4, 2))
        self.assertEqual(windows["Y"].shape, (6, 3, 2))
        self.assertEqual(windows["timestamp"].tolist(), [4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(windows["X"][0], run_values[:4])
        np.testing.assert_array_equal(windows["Y"][0], run_values[4:7])

    def test_build_windows_returns_empty_for_too_short_run(self) -> None:
        run_values = np.zeros((5, 53), dtype=np.float32)
        time_index = np.arange(5, dtype=np.int32)

        windows = build_windows(
            run_values=run_values,
            time_index=time_index,
            window_length=4,
            horizon=2,
        )

        self.assertEqual(windows["X"].shape, (0, 4, 53))
        self.assertEqual(windows["Y"].shape, (0, 2, 53))
        self.assertEqual(windows["timestamp"].shape, (0,))

    def test_aggregate_unit_scores_excludes_feed_side_from_five_unit_scores(self) -> None:
        variable_scores = np.array(
            [
                [10.0, 20.0, 30.0, 40.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        )
        unit_scores, feed_scores = aggregate_unit_scores(
            variable_scores,
            unit_order=["A", "B"],
            unit_indices={
                "A": np.array([1, 2], dtype=np.int64),
                "B": np.array([3], dtype=np.int64),
            },
            feed_side_indices=np.array([0], dtype=np.int64),
            member_reduce="mean",
        )

        np.testing.assert_allclose(unit_scores, np.array([[25.0, 40.0], [2.5, 4.0]], dtype=np.float32))
        np.testing.assert_allclose(feed_scores, np.array([10.0, 1.0], dtype=np.float32))

    def test_compute_run_metrics_requires_truth_inclusion(self) -> None:
        metrics = compute_run_metrics(
            ranking=["Reactor", "Stripper", "Separator"],
            early_ranking=["Stripper", "Reactor", "Separator"],
            truth_row={
                "included_in_main_eval": True,
                "primary_unit": "Reactor",
                "expected_units": ["Reactor", "Separator"],
            },
        )
        self.assertEqual(metrics["top1"], 1.0)
        self.assertEqual(metrics["top3"], 1.0)
        self.assertEqual(metrics["soft_precision_at_3"], 2.0 / 2.0)
        self.assertEqual(metrics["early_hit"], 0.0)

        skipped = compute_run_metrics(
            ranking=["Reactor", "Stripper", "Separator"],
            early_ranking=["Stripper", "Reactor", "Separator"],
            truth_row={
                "included_in_main_eval": False,
                "primary_unit": "Reactor",
                "expected_units": ["Reactor", "Separator"],
            },
        )
        self.assertTrue(all(value is None for value in skipped.values()))

    def test_select_fault_runs_merges_run_lookup_fields(self) -> None:
        selected = select_fault_runs(
            fault_eval_manifest={
                "runs": [
                    {"run_key": "M4_d06", "scenario": "d06", "mode": "M4"},
                    {"run_key": "M4_d01", "scenario": "d01", "mode": "M4"},
                ]
            },
            run_lookup={
                "M4_d01": {"run_key": "M4_d01", "time_index_file": "runs/M4_d01_time_index.npy"},
                "M4_d06": {"run_key": "M4_d06", "time_index_file": "runs/M4_d06_time_index.npy"},
            },
            profile_cfg={"run_key_allowlist": ["M4_d06", "M4_d01"]},
        )

        self.assertEqual([entry["run_key"] for entry in selected], ["M4_d01", "M4_d06"])
        self.assertEqual(selected[0]["time_index_file"], "runs/M4_d01_time_index.npy")
        self.assertEqual(selected[1]["time_index_file"], "runs/M4_d06_time_index.npy")

    def test_canonical_expected_units_deduplicates_primary(self) -> None:
        units = canonical_expected_units("Reactor", ["Separator", "Reactor", "Separator"])
        self.assertEqual(set(units), {"Reactor", "Separator"})
        self.assertEqual(len(units), 2)

    def test_truth_row_selection_counts_only_main_eval_rows(self) -> None:
        self.assertTrue(
            truth_row_is_evaluable(
                {
                    "included_in_main_eval": True,
                    "primary_unit": "Reactor",
                    "expected_units": ["Reactor"],
                }
            )
        )
        self.assertFalse(
            truth_row_is_evaluable(
                {
                    "included_in_main_eval": False,
                    "primary_unit": "Reactor",
                    "expected_units": ["Reactor"],
                }
            )
        )
        count = count_selected_evaluable_runs(
            [
                {"scenario": "d01"},
                {"scenario": "d02"},
                {"scenario": "d03"},
            ],
            {
                "d01": {"included_in_main_eval": True, "primary_unit": "Condenser", "expected_units": ["Condenser"]},
                "d02": {"included_in_main_eval": True, "primary_unit": "Reactor", "expected_units": ["Reactor"]},
                "d03": {"included_in_main_eval": False, "primary_unit": None, "expected_units": []},
            },
        )
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
