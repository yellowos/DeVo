"""Smoke tests for experiments-layer shared execution/result helpers."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments import (
    ArtifactPathRef,
    create_run_handle,
    load_experiment_config,
    scan_run_results,
    summarize_to_csv,
)
from methods.base import ArtifactRef, MethodResult


class ExperimentsLayerSmokeTest(unittest.TestCase):
    def _write_dummy_success_result(self, results_root: Path, root: Path) -> Path:
        run = create_run_handle(
            experiment_name="acceptance_smoke",
            dataset="toyset",
            method="dummy_method",
            seed=1,
            config={"epochs": 1},
            results_root=results_root,
            run_id="success_run",
        )
        return run.save_success(
            metrics={"test.rmse": {"value": 0.5, "split": "test", "higher_is_better": False}},
            artifact_paths={
                "predictions": ArtifactPathRef(
                    kind="predictions",
                    path=root / "artifacts" / "predictions.npy",
                )
            },
        )

    def test_dummy_run_can_be_written_scanned_and_summarized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "dummy_config.json"
            config_path.write_text(
                json.dumps({"epochs": 3, "optimizer": {"lr": 0.01}}, indent=2),
                encoding="utf-8",
            )

            config = load_experiment_config(config_path, overrides={"optimizer": {"weight_decay": 0.0}})
            self.assertEqual(config["optimizer"]["lr"], 0.01)
            self.assertEqual(config["optimizer"]["weight_decay"], 0.0)

            method_result = MethodResult(
                model_state_path=str(root / "artifacts" / "dummy_state.json"),
                artifacts={
                    "weights": ArtifactRef(kind="checkpoint", path=root / "artifacts" / "weights.pt"),
                },
            )

            success_run = create_run_handle(
                experiment_name="nonlinear_public_smoke",
                dataset="silverbox",
                method="dummy_method",
                seed=7,
                config=config,
                results_root=root / "results",
                run_id="success_run",
            )
            success_path = success_run.save_success(
                metrics={
                    "test.rmse": {
                        "value": 0.125,
                        "split": "test",
                        "higher_is_better": False,
                    },
                    "val.loss": 0.2,
                },
                method_result=method_result,
                artifact_paths={
                    "predictions": ArtifactPathRef(
                        kind="predictions",
                        path=root / "artifacts" / "predictions.npy",
                    )
                },
            )
            self.assertTrue(success_path.exists())

            skipped_run = create_run_handle(
                experiment_name="nonlinear_public_smoke",
                dataset="silverbox",
                method="dummy_method",
                seed=8,
                config=config,
                results_root=root / "results",
                run_id="skipped_run",
            )
            skipped_run.save_skipped("dataset/method pair intentionally skipped")

            failed_run = create_run_handle(
                experiment_name="nonlinear_public_smoke",
                dataset="silverbox",
                method="dummy_method",
                seed=9,
                config=config,
                results_root=root / "results",
                run_id="failed_run",
            )
            try:
                raise RuntimeError("dummy failure")
            except RuntimeError as exc:
                failed_run.save_failed(exc=exc)

            records = scan_run_results(root / "results")
            self.assertEqual(len(records), 3)
            states = {record.run_id: record.status.state for record in records}
            self.assertEqual(states["success_run"], "success")
            self.assertEqual(states["skipped_run"], "skipped")
            self.assertEqual(states["failed_run"], "failed")

            success_record = next(record for record in records if record.run_id == "success_run")
            self.assertIn("weights", success_record.artifact_paths)
            self.assertIn("model_state", success_record.artifact_paths)
            self.assertEqual(success_record.metrics["test.rmse"].split, "test")

            csv_path = summarize_to_csv(records, root / "results" / "summary.csv", results_root=root / "results")
            self.assertTrue(csv_path.exists())
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 3)
            rows_by_id = {row["run_id"]: row for row in rows}
            self.assertEqual(rows_by_id["success_run"]["status.state"], "success")
            self.assertEqual(rows_by_id["skipped_run"]["status.state"], "skipped")
            self.assertEqual(rows_by_id["failed_run"]["status.state"], "failed")
            self.assertEqual(rows_by_id["success_run"]["metrics.test.rmse.value"], "0.125")
            self.assertTrue(rows_by_id["success_run"]["artifact_paths.model_state.path"].endswith("dummy_state.json"))
            self.assertTrue(rows_by_id["success_run"]["result_path"].endswith("result.json"))
            self.assertTrue(rows_by_id["success_run"]["run_dir"].endswith("success_run"))

    def test_collect_cli_writes_result_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            results_root = root / "results"
            self._write_dummy_success_result(results_root, root)
            output_path = root / "summary.csv"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.collect",
                    str(results_root),
                    "--output",
                    str(output_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertTrue(output_path.exists())
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertTrue(row["result_path"])
            self.assertTrue(row["run_dir"])
            self.assertTrue(row["result_path"].endswith("result.json"))
            self.assertTrue(row["run_dir"].endswith("success_run"))

    def test_collect_script_help_succeeds(self):
        script_path = Path(__file__).resolve().parents[2] / "experiments" / "collect.py"
        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("python -m experiments.collect", completed.stdout)


if __name__ == "__main__":
    unittest.main()
