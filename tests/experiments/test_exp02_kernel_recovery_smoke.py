"""Fast acceptance smoke for exp02 kernel recovery scripts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class Exp02KernelRecoverySmokeTest(unittest.TestCase):
    def test_experiment_modules_import(self) -> None:
        import experiments.nonlinear.exp02_kernel_recovery.collect as collect
        import experiments.nonlinear.exp02_kernel_recovery.plot as plot
        import experiments.nonlinear.exp02_kernel_recovery.run as run

        self.assertEqual(run.EXPERIMENT_NAME, "exp02_kernel_recovery")
        self.assertTrue(callable(collect.collect))
        self.assertTrue(callable(plot.plot))

    def test_collect_and_plot_on_fake_run_dir(self) -> None:
        from experiments.nonlinear.exp02_kernel_recovery.collect import collect
        from experiments.nonlinear.exp02_kernel_recovery.plot import plot

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            for dataset_name, metric_name in (
                ("volterra_wiener", "knmse"),
                ("duffing", "gfrf_re"),
            ):
                result_dir = run_dir / dataset_name / "devo"
                result_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "dataset_name": dataset_name,
                    "method_name": "devo",
                    "status": "warning",
                    "recovery_status": "completed",
                    "metric_status": "failed",
                    "metric_name": metric_name,
                    "metrics": {
                        metric_name: {
                            "metric_name": metric_name,
                            "status": "failed",
                            "value": None,
                            "message": "placeholder truth is missing",
                        }
                    },
                    "artifacts": {
                        "recovered_kernel_manifest": str(result_dir / "recovered_kernels_manifest.json"),
                        "truth_reference_path": None,
                        "truth_payload_path": None,
                        "recovered_gfrf_npz": None,
                    },
                }
                (result_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

            collected = collect(run_dir, output_dir=run_dir / "collected")
            plotted = plot(run_dir, output_dir=run_dir / "plots")

            self.assertTrue(Path(collected["paper_table_csv"]).exists())
            self.assertTrue(Path(collected["paper_table_md"]).exists())
            self.assertTrue(Path(plotted["plots"]["knmse_comparison"]).exists())
            self.assertTrue(Path(plotted["plots"]["gfrf_re_comparison"]).exists())
            self.assertIsNone(plotted["plots"]["duffing_gfrf_comparison"])

    def test_truth_loaders_accept_materialized_reference_payloads(self) -> None:
        from experiments.nonlinear.exp02_kernel_recovery.run import load_truth_gfrf, load_truth_kernel

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            kernel_payload = tmp_path / "kernel_truth.npz"
            gfrf_payload = tmp_path / "gfrf_truth.npz"
            kernel_ref = tmp_path / "kernel_reference.json"
            gfrf_ref = tmp_path / "gfrf_reference.json"

            np.savez_compressed(kernel_payload, order_1=np.asarray([0.1, 0.2, 0.3], dtype=np.float64))
            np.savez_compressed(gfrf_payload, order_1=np.asarray([1.0 + 0.0j, 0.5 + 0.25j]))
            kernel_ref.write_text(
                json.dumps({"benchmark_name": "volterra_wiener", "kernel_coefficients_path": str(kernel_payload)}),
                encoding="utf-8",
            )
            gfrf_ref.write_text(
                json.dumps({"benchmark_name": "duffing", "gfrf_coefficients_path": str(gfrf_payload)}),
                encoding="utf-8",
            )

            bundle = SimpleNamespace(
                artifacts=SimpleNamespace(
                    truth_file=None,
                    extra={
                        "kernel_reference_file": str(kernel_ref),
                        "gfrf_reference_file": str(gfrf_ref),
                    },
                ),
                meta=SimpleNamespace(extras={}),
            )

            kernel_truth = load_truth_kernel(bundle)
            gfrf_truth = load_truth_gfrf(bundle)

            self.assertEqual(kernel_truth["status"], "completed")
            self.assertEqual(gfrf_truth["status"], "completed")
            self.assertIn(1, kernel_truth["orders"])
            self.assertIn(1, gfrf_truth["orders"])


if __name__ == "__main__":
    unittest.main()
