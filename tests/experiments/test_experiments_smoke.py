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
    compute_config_hash,
    create_run_handle,
    load_experiment_config,
    scan_run_results,
    summarize_to_csv,
    write_resolved_config,
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

    def test_config_deep_merge_preserves_unmodified_nested_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base = {
                "model": {"name": "devo", "orders": [1, 2], "hidden": {"width": 32}},
                "training": {"batch_size": 16, "epochs": 2},
                "data": {"dataset": "toy", "window_length": 8},
            }
            config = load_experiment_config(base, overrides={"training": {"batch_size": 64}})
            self.assertEqual(config["training"]["batch_size"], 64)
            self.assertEqual(config["training"]["epochs"], 2)
            self.assertEqual(config["model"]["hidden"]["width"], 32)
            self.assertEqual(config["data"]["window_length"], 8)

            path_a = write_resolved_config(config, root / "resolved_a.json")
            path_b = write_resolved_config(config, root / "resolved_b.json")
            hash_a = compute_config_hash(config)
            hash_b = compute_config_hash(load_experiment_config(config))
            self.assertEqual(path_a.read_text(encoding="utf-8"), path_b.read_text(encoding="utf-8"))
            self.assertEqual(hash_a, hash_b)

    def test_run_id_uniqueness_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = create_run_handle(
                experiment_name="repeatable",
                dataset="toy",
                method="dummy",
                seed=1,
                config={"epochs": 1},
                results_root=root / "results",
            )
            second = create_run_handle(
                experiment_name="repeatable",
                dataset="toy",
                method="dummy",
                seed=1,
                config={"epochs": 1},
                results_root=root / "results",
            )
            self.assertNotEqual(first.paths.run_id, second.paths.run_id)
            self.assertNotEqual(first.paths.run_dir, second.paths.run_dir)
            first.save_success(metrics={"test.rmse": 0.1})
            second.save_success(metrics={"test.rmse": 0.2})
            self.assertTrue(first.paths.result_path.exists())
            self.assertTrue(second.paths.result_path.exists())

    def test_collect_preserves_multiple_same_name_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            results_root = root / "results"
            first = create_run_handle(
                experiment_name="repeatable",
                dataset="toy",
                method="dummy",
                seed=2,
                config={"epochs": 1},
                results_root=results_root,
            )
            second = create_run_handle(
                experiment_name="repeatable",
                dataset="toy",
                method="dummy",
                seed=2,
                config={"epochs": 1},
                results_root=results_root,
            )
            first.save_success(metrics={"test.rmse": 0.1})
            second.save_success(metrics={"test.rmse": 0.2})

            csv_path = root / "summary.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.collect",
                    str(results_root),
                    "--output",
                    str(csv_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertNotEqual(rows[0]["run_id"], rows[1]["run_id"])

    def test_collect_ignores_non_whitelisted_dirs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            valid = create_run_handle(
                experiment_name="whitelist",
                dataset="toy",
                method="dummy",
                seed=0,
                config={"epochs": 1},
                results_root=root / "results",
            )
            valid.save_success(metrics={"test.rmse": 0.3})

            invalid_dir = root / "results" / "whitelist" / "toy" / "dummy" / "seed_0" / "invalid_manual"
            invalid_dir.mkdir(parents=True, exist_ok=True)
            (invalid_dir / "result.json").write_text(json.dumps({"run_id": "invalid"}), encoding="utf-8")

            records = scan_run_results(root / "results")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].run_id, valid.paths.run_id)


if __name__ == "__main__":
    unittest.main()
