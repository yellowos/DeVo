"""End-to-end hydraulic builder verification."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.builders.build_hydraulic import run_build
from data.adapters.hydraulic_adapter import HydraulicAdapter
from data.checks.unified_data_layer_checks import validate_data_layer


class HydraulicBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = ROOT
        run_build(cls.root)
        cls.processed_root = cls.root / "data" / "processed" / "hydraulic"
        cls.split_root = cls.root / "data" / "splits" / "hydraulic"
        cls.metadata_root = cls.root / "data" / "metadata" / "hydraulic"

    def test_processed_manifest_and_artifacts_exist(self) -> None:
        manifest_path = self.processed_root / "hydraulic_processed_manifest.json"
        representations_path = self.processed_root / "hydraulic_representations_manifest.json"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(representations_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        representations = json.loads(representations_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["representation_name"], "cycle_60")
        self.assertEqual(manifest["cycle_tensor_shape"], [2205, 60, 17])
        self.assertEqual(representations["default_representation"], "cycle_60")
        self.assertIn("cycle_60", representations["representations"])
        self.assertIn("cycle_600", representations["representations"])
        self.assertTrue((self.metadata_root / "channel_map.json").exists())
        self.assertTrue((self.metadata_root / "subsystem_groups.json").exists())
        self.assertTrue((self.metadata_root / "single_fault_protocol.json").exists())
        self.assertTrue((self.split_root / "hydraulic_split_manifest.json").exists())
        self.assertTrue((self.processed_root / "cycle_60" / "hydraulic_processed_manifest.json").exists())
        self.assertTrue((self.processed_root / "cycle_600" / "hydraulic_processed_manifest.json").exists())

    def test_both_representations_have_expected_shapes(self) -> None:
        cycle_60_manifest = json.loads(
            (self.processed_root / "cycle_60" / "hydraulic_processed_manifest.json").read_text(encoding="utf-8")
        )
        cycle_600_manifest = json.loads(
            (self.processed_root / "cycle_600" / "hydraulic_processed_manifest.json").read_text(encoding="utf-8")
        )

        self.assertEqual(cycle_60_manifest["cycle_tensor_shape"], [2205, 60, 17])
        self.assertEqual(cycle_600_manifest["cycle_tensor_shape"], [2205, 600, 17])
        self.assertEqual(cycle_60_manifest["target_cycle_length"], 60)
        self.assertEqual(cycle_600_manifest["target_cycle_length"], 600)
        self.assertEqual(cycle_60_manifest["resampling"]["channels"]["PS1"]["transform"], "mean_pool")
        self.assertEqual(cycle_60_manifest["resampling"]["channels"]["PS1"]["factor"], 100)
        self.assertEqual(cycle_60_manifest["resampling"]["channels"]["FS1"]["transform"], "mean_pool")
        self.assertEqual(cycle_60_manifest["resampling"]["channels"]["FS1"]["factor"], 10)
        self.assertEqual(cycle_60_manifest["resampling"]["channels"]["TS1"]["transform"], "identity")
        self.assertEqual(cycle_600_manifest["resampling"]["channels"]["PS1"]["transform"], "mean_pool")
        self.assertEqual(cycle_600_manifest["resampling"]["channels"]["PS1"]["factor"], 10)
        self.assertEqual(cycle_600_manifest["resampling"]["channels"]["FS1"]["transform"], "identity")
        self.assertEqual(cycle_600_manifest["resampling"]["channels"]["TS1"]["transform"], "repeat_hold")
        self.assertEqual(cycle_600_manifest["resampling"]["channels"]["TS1"]["factor"], 10)

    def test_derived_protocol_fields_cover_expected_fault_subsystems(self) -> None:
        label_paths = [
            self.processed_root / "cycle_60" / "hydraulic_cycle_labels.csv",
            self.processed_root / "cycle_600" / "hydraulic_cycle_labels.csv",
        ]
        rows_per_version = []
        for label_path in label_paths:
            with label_path.open("r", encoding="utf-8", newline="") as f:
                rows_per_version.append(list(csv.DictReader(f)))

        rows = rows_per_version[0]
        self.assertEqual(rows, rows_per_version[1])

        self.assertEqual(len(rows), 2205)
        self.assertIn("is_healthy", rows[0])
        self.assertIn("is_single_component_fault", rows[0])
        self.assertIn("fault_subsystem", rows[0])
        self.assertIn("stable_flag", rows[0])

        healthy_count = sum(int(row["is_healthy"]) for row in rows)
        single_fault_count = sum(int(row["is_single_component_fault"]) for row in rows)
        fault_subsystems = {row["fault_subsystem"] for row in rows if row["fault_subsystem"]}

        self.assertGreater(healthy_count, 0)
        self.assertGreater(single_fault_count, 0)
        self.assertEqual(fault_subsystems, {"Cooler", "Valve", "Pump", "Accumulator"})

    def test_hydraulic_data_layer_validation_passes(self) -> None:
        report = validate_data_layer(project_root=self.root, task_family="hydraulic")
        self.assertEqual(report.errors, [])

    def test_hydraulic_adapter_can_load_processed_bundle(self) -> None:
        bundle = HydraulicAdapter.load_processed_bundle(self.processed_root)
        self.assertEqual(bundle.meta.dataset_name, "hydraulic")
        self.assertEqual(bundle.meta.input_dim, 17)
        self.assertEqual(bundle.meta.window_length, 60)

        bundle_600 = HydraulicAdapter.load_processed_bundle(self.processed_root, representation="cycle_600")
        self.assertEqual(bundle_600.meta.window_length, 600)


if __name__ == "__main__":
    unittest.main()
