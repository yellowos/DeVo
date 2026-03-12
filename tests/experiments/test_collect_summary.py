"""Tests for the experiments-layer summary collector."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.common.summary import collect_experiment_summary, write_summary_outputs


class CollectSummaryTest(unittest.TestCase):
    def test_current_repo_surfaces_missing_and_not_ready_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiments_root = Path(tmp_dir) / "experiments"
            output_dir = Path(tmp_dir) / "summary"
            report = collect_experiment_summary(
                project_root=ROOT,
                experiments_root=experiments_root,
                output_dir=output_dir,
            )

        by_id = {record.experiment_id: record for record in report.experiments}
        self.assertEqual(by_id["nonlinear/exp01_prediction_benchmark"].overall_status, "missing")
        self.assertEqual(by_id["nonlinear/exp02_kernel_recovery"].overall_status, "missing")
        self.assertEqual(by_id["hydraulic/exp01_unsupervised_subsystem_fault_isolation"].overall_status, "not_ready")
        self.assertEqual(by_id["tep/exp01_five_unit_fault_isolation"].overall_status, "not_ready")
        self.assertEqual(by_id["tep/exp02_fault_propagation_analysis"].overall_status, "not_ready")
        self.assertIn(
            "curated",
            " ".join(by_id["tep/exp01_five_unit_fault_isolation"].dependency_notes).lower(),
        )

    def test_detects_ready_partial_and_skipped_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiments_root = Path(tmp_dir) / "experiments"
            output_dir = Path(tmp_dir) / "summary"

            exp01 = experiments_root / "nonlinear" / "exp01_prediction_benchmark"
            exp01.mkdir(parents=True)
            (exp01 / "summary.json").write_text(
                json.dumps({"status": "ready", "metrics": {"rmse": 0.123}}, indent=2),
                encoding="utf-8",
            )
            (exp01 / "leaderboard.csv").write_text("method,rmse\ndevo,0.123\n", encoding="utf-8")
            (exp01 / "tables").mkdir()
            (exp01 / "tables" / "prediction_benchmark.tex").write_text(
                "\\begin{tabular}{ll}\nmethod & rmse \\\\\n\\end{tabular}\n",
                encoding="utf-8",
            )
            (exp01 / "figures").mkdir()
            (exp01 / "figures" / "prediction_overview.png").write_bytes(b"png")

            exp02 = experiments_root / "nonlinear" / "exp02_kernel_recovery"
            exp02.mkdir(parents=True)
            (exp02 / "summary.json").write_text(
                json.dumps({"status": "running", "metrics": {"mae": 0.456}}, indent=2),
                encoding="utf-8",
            )

            exp03 = experiments_root / "nonlinear" / "exp03_ablation_hparam"
            exp03.mkdir(parents=True)
            (exp03 / "SKIPPED.md").write_text("status: skipped\n", encoding="utf-8")

            report = collect_experiment_summary(
                project_root=ROOT,
                experiments_root=experiments_root,
                output_dir=output_dir,
            )
            outputs = write_summary_outputs(report, output_dir=output_dir)

            by_id = {record.experiment_id: record for record in report.experiments}
            self.assertEqual(by_id["nonlinear/exp01_prediction_benchmark"].overall_status, "ready")
            self.assertEqual(by_id["nonlinear/exp01_prediction_benchmark"].paper_items_ready, 1)
            self.assertEqual(by_id["nonlinear/exp02_kernel_recovery"].overall_status, "partial")
            self.assertEqual(by_id["nonlinear/exp03_ablation_hparam"].overall_status, "skipped")

            for path in outputs.values():
                self.assertTrue(path.exists(), msg=str(path))

            with outputs["artifact_manifest"].open("r", encoding="utf-8", newline="") as handle:
                artifact_rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["artifact_kind"] == "table" for row in artifact_rows))
            self.assertTrue(any(row["artifact_kind"] == "figure" for row in artifact_rows))

            with outputs["paper_tables"].open("r", encoding="utf-8", newline="") as handle:
                paper_rows = list(csv.DictReader(handle))
            exp01_row = next(row for row in paper_rows if row["experiment_id"] == "nonlinear/exp01_prediction_benchmark")
            self.assertEqual(exp01_row["status"], "ready")


if __name__ == "__main__":
    unittest.main()
