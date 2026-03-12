from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.adapters.tep_adapter import TEPAdapter
from data.builders.build_tep import run_build
from data.checks.unified_data_layer_checks import validate_data_layer


class TEPBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = ROOT
        run_build(cls.root)
        cls.processed_root = cls.root / "data" / "processed" / "tep"
        cls.metadata_root = cls.root / "data" / "metadata" / "tep"
        cls.manifest = json.loads((cls.processed_root / "tep_processed_manifest.json").read_text(encoding="utf-8"))
        cls.run_manifest = json.loads((cls.processed_root / "tep_run_manifest.json").read_text(encoding="utf-8"))
        cls.fault_eval_manifest = json.loads((cls.processed_root / "tep_fault_eval_manifest.json").read_text(encoding="utf-8"))

    def test_processed_manifests_and_metadata_exist(self) -> None:
        self.assertTrue((self.processed_root / "tep_processed_manifest.json").exists())
        self.assertTrue((self.processed_root / "tep_run_manifest.json").exists())
        self.assertTrue((self.root / "data" / "splits" / "tep" / "tep_mode_holdout_split_manifest.json").exists())
        for name in (
            "mode_holdout_protocol.json",
            "five_unit_definition.json",
            "feed_side_definition.json",
            "fault_truth_table.json",
            "scenario_to_idv_map.json",
        ):
            self.assertTrue((self.metadata_root / name).exists(), msg=f"missing {name}")

    def test_run_manifest_parses_all_runs_and_keeps_variable_length_faults(self) -> None:
        runs = self.run_manifest["runs"]
        self.assertEqual(len(runs), 174)
        normal_runs = [run for run in runs if not run["is_fault"]]
        fault_runs = [run for run in runs if run["is_fault"]]
        self.assertEqual({run["n_steps"] for run in normal_runs}, {7201})
        self.assertIn(("M4", "d01", 631), {(run["mode"], run["scenario"], run["n_steps"]) for run in fault_runs})
        self.assertIn(("M1", "d06", 712), {(run["mode"], run["scenario"], run["n_steps"]) for run in fault_runs})
        self.assertIn(("M4", "d07", 156), {(run["mode"], run["scenario"], run["n_steps"]) for run in fault_runs})
        self.assertIn(("M3", "d08", 5682), {(run["mode"], run["scenario"], run["n_steps"]) for run in fault_runs})
        self.assertIn(("M2", "d13", 5470), {(run["mode"], run["scenario"], run["n_steps"]) for run in fault_runs})

    def test_windowed_normal_splits_keep_mode_holdout_contract(self) -> None:
        bundle = TEPAdapter.load_processed_bundle(self.processed_root)
        self.assertEqual(bundle.meta.input_dim, 53)
        self.assertEqual(bundle.meta.output_dim, 53)
        self.assertEqual(set(bundle.train.meta["mode"].tolist()), {"M1", "M2", "M3", "M4"})
        self.assertEqual(set(bundle.val.meta["mode"].tolist()), {"M5"})
        self.assertEqual(set(bundle.test.meta["mode"].tolist()), {"M6"})
        self.assertEqual(set(bundle.train.meta["scenario_id"].tolist()), {"d00"})
        self.assertEqual(set(bundle.val.meta["scenario_id"].tolist()), {"d00"})
        self.assertEqual(set(bundle.test.meta["scenario_id"].tolist()), {"d00"})
        self.assertEqual(set(bundle.train.meta["fault_id"].tolist()), {"healthy"})
        self.assertEqual(set(bundle.val.meta["fault_id"].tolist()), {"healthy"})
        self.assertEqual(set(bundle.test.meta["fault_id"].tolist()), {"healthy"})

    def test_fault_eval_manifest_uses_raw_dxx_namespace(self) -> None:
        rows = self.fault_eval_manifest["runs"]
        self.assertEqual(len(rows), 168)
        scenarios = {row["scenario"] for row in rows}
        self.assertEqual(scenarios, {f"d{i:02d}" for i in range(1, 29)})
        self.assertTrue(all(str(row["idv"]).startswith("idv") for row in rows))

    def test_unified_tep_checks_pass(self) -> None:
        report = validate_data_layer(project_root=self.root, task_family="tep")
        self.assertEqual(report.errors, [])


if __name__ == "__main__":
    unittest.main()
