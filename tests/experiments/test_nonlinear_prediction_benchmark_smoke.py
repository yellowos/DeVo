from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "nonlinear" / "exp01_prediction_benchmark"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NonlinearPredictionBenchmarkExperimentSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_module = _load_module("exp01_run_module", EXPERIMENT_DIR / "run.py")
        cls.collect_module = _load_module("exp01_collect_module", EXPERIMENT_DIR / "collect.py")
        cls.plot_module = _load_module("exp01_plot_module", EXPERIMENT_DIR / "plot.py")
        cls.config_path = EXPERIMENT_DIR / "config.yaml"

    def test_experiment_scripts_import_and_smoke_profile_resolves(self) -> None:
        config = self.run_module._resolve_config(self.config_path, "smoke")
        self.assertEqual(config["experiment_name"], "exp01_prediction_benchmark")
        self.assertEqual(config["datasets"], ["coupled_duffing", "cascaded_tanks"])
        self.assertEqual(config["methods"], ["narmax", "mlp", "devo"])
        self.assertEqual(config["seeds"], [0])

    def test_metrics_align_flattened_predictions(self) -> None:
        metrics = self.run_module._compute_metrics(
            [[[1.0]], [[3.0]]],
            [[1.0], [2.0]],
        )
        self.assertAlmostEqual(metrics["mse"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], 0.5**0.5)
        self.assertAlmostEqual(metrics["nmse"], 0.5)

    def test_exception_classification_only_skips_explicit_input_contract_errors(self) -> None:
        self.assertEqual(
            self.run_module._classify_exception(
                ValueError("X window_length mismatch: expected 128, got 64.")
            ),
            "skipped",
        )
        self.assertEqual(
            self.run_module._classify_exception(
                ValueError("Prediction shape mismatch: expected flattened shape (64, 1), got (64, 2).")
            ),
            "failed",
        )
        self.assertEqual(
            self.run_module._classify_exception(
                RuntimeError("Expected input_dim=1 after layer reshape, got 2.")
            ),
            "failed",
        )

    def test_collect_and_plot_helpers_work_from_summary_csv(self) -> None:
        rows = [
            {
                "dataset": "coupled_duffing",
                "method": "narmax",
                "status": "success",
                "metrics": {"nmse": 1.5, "rmse": 0.5},
            },
            {
                "dataset": "coupled_duffing",
                "method": "narmax",
                "status": "success",
                "metrics": {"nmse": 2.5, "rmse": 0.75},
            },
            {
                "dataset": "cascaded_tanks",
                "method": "mlp",
                "status": "failed",
                "metrics": None,
            },
        ]
        aggregated = self.collect_module._aggregate(
            rows,
            ["coupled_duffing", "cascaded_tanks"],
            ["narmax", "mlp"],
        )
        lookup = {(row["dataset"], row["method"]): row for row in aggregated}
        self.assertAlmostEqual(lookup[("coupled_duffing", "narmax")]["nmse_mean"], 2.0)
        self.assertEqual(lookup[("cascaded_tanks", "mlp")]["num_failed"], 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "benchmark_summary_long.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "dataset",
                        "method",
                        "num_runs",
                        "num_success",
                        "num_failed",
                        "num_skipped",
                        "nmse_mean",
                        "nmse_std",
                        "rmse_mean",
                        "rmse_std",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "coupled_duffing",
                        "method": "narmax",
                        "num_runs": 2,
                        "num_success": 2,
                        "num_failed": 0,
                        "num_skipped": 0,
                        "nmse_mean": 2.0,
                        "nmse_std": 0.5,
                        "rmse_mean": 0.625,
                        "rmse_std": 0.125,
                    }
                )
            loaded_rows = self.plot_module._load_summary_rows(csv_path)
            self.assertEqual(len(loaded_rows), 1)
            self.assertEqual(loaded_rows[0]["dataset"], "coupled_duffing")


if __name__ == "__main__":
    unittest.main()
