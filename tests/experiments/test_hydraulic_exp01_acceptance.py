"""Minimal acceptance checks for the hydraulic exp01 experiment scripts."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class HydraulicExp01AcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_module = importlib.import_module(
            "experiments.hydraulic.exp01_unsupervised_subsystem_fault_isolation.run"
        )
        cls.collect_module = importlib.import_module(
            "experiments.hydraulic.exp01_unsupervised_subsystem_fault_isolation.collect"
        )
        cls.plot_module = importlib.import_module(
            "experiments.hydraulic.exp01_unsupervised_subsystem_fault_isolation.plot"
        )
        cls.config_path = (
            REPO_ROOT
            / "experiments"
            / "hydraulic"
            / "exp01_unsupervised_subsystem_fault_isolation"
            / "config.yaml"
        )

    def test_modules_import(self) -> None:
        self.assertTrue(hasattr(self.run_module, "main"))
        self.assertTrue(hasattr(self.collect_module, "main"))
        self.assertTrue(hasattr(self.plot_module, "main"))

    def test_hydraulic_data_ready_for_main_experiment(self) -> None:
        config = self.run_module._load_yaml(self.config_path)
        data_bundle, error, diagnostics = self.run_module._load_hydraulic_data(config)
        self.assertIsNone(error)
        self.assertIsNotNone(data_bundle)
        self.assertEqual(len(data_bundle.channel_names), 17)
        self.assertGreater(len(data_bundle.healthy_records), 0)
        self.assertGreater(len(data_bundle.eval_records), 0)
        self.assertEqual(set(data_bundle.subsystem_indices), {"Cooler", "Valve", "Pump", "Accumulator"})
        self.assertEqual(diagnostics["representation"], "cycle_60")

    def test_missing_processed_root_is_reported_as_not_ready(self) -> None:
        config = self.run_module._load_yaml(self.config_path)
        config["experiment"]["processed_root"] = "data/processed/hydraulic_missing_for_test"
        data_bundle, error, diagnostics = self.run_module._load_hydraulic_data(config)
        self.assertIsNone(data_bundle)
        self.assertIsNotNone(error)
        self.assertIn("not loadable", error)
        self.assertIn("hydraulic_missing_for_test", diagnostics["processed_root"])

    def test_collect_handles_skipped_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            method_dir = run_dir / "methods" / "devo"
            method_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": "skipped",
                "method": "devo",
                "reason": "data not ready",
                "metrics": {
                    "num_eval_samples": 0,
                    "top1_isolation_accuracy": None,
                    "top2_coverage": None,
                    "mean_rank_true_subsystem": None,
                    "winning_margin": None,
                },
                "subsystem_hit_rates": {
                    name: {"num_samples": 0, "num_top1_hits": 0, "top1_hit_rate": None}
                    for name in ("Cooler", "Valve", "Pump", "Accumulator")
                },
            }
            (method_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

            argv_backup = sys.argv[:]
            try:
                sys.argv = ["collect.py", "--run-dir", str(run_dir)]
                exit_code = self.collect_module.main()
            finally:
                sys.argv = argv_backup

            self.assertEqual(exit_code, 0)
            main_table = (run_dir / "collected" / "table_5_2_5_main.csv").read_text(encoding="utf-8")
            self.assertIn("devo,skipped,data not ready,0", main_table)

    def test_plot_renders_from_minimal_collected_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            collected_dir = run_dir / "collected"
            collected_dir.mkdir(parents=True, exist_ok=True)

            (collected_dir / "table_5_2_5_main.csv").write_text(
                "\n".join(
                    [
                        "method,status,reason,num_eval_samples,top1_isolation_accuracy,top2_coverage,mean_rank_true_subsystem,winning_margin,result_path",
                        "devo,completed,,1,1.0,1.0,1.0,0.1,/tmp/devo.json",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (collected_dir / "table_5_2_6_subsystems_long.csv").write_text(
                "\n".join(
                    [
                        "method,status,reason,subsystem,num_samples,num_top1_hits,top1_hit_rate",
                        "devo,completed,,Cooler,1,1,1.0",
                        "devo,completed,,Valve,0,0,",
                        "devo,completed,,Pump,0,0,",
                        "devo,completed,,Accumulator,0,0,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv_backup = sys.argv[:]
            try:
                sys.argv = ["plot.py", "--run-dir", str(run_dir)]
                exit_code = self.plot_module.main()
            finally:
                sys.argv = argv_backup

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "plots" / "top1_top2_comparison.png").exists())
            self.assertTrue((run_dir / "plots" / "subsystem_top1_hit_rates.png").exists())


if __name__ == "__main__":
    unittest.main()
