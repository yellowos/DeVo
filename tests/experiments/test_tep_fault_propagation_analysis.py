from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class TEPFaultPropagationAnalysisTest(unittest.TestCase):
    def test_modules_import(self) -> None:
        for name in (
            "experiments.tep.exp02_fault_propagation_analysis.run",
            "experiments.tep.exp02_fault_propagation_analysis.collect",
            "experiments.tep.exp02_fault_propagation_analysis.plot",
        ):
            module = importlib.import_module(name)
            self.assertTrue(hasattr(module, "main"), msg=f"{name} missing main()")

    def test_effective_run_scope_truncates_short_run_without_padding(self) -> None:
        from experiments.tep.exp02_fault_propagation_analysis.run import effective_run_scope

        scope, indices, note = effective_run_scope(
            count=28,
            min_windows_to_process=16,
            short_run_policy="truncate",
            early_window_cap=256,
            full_window_count=7073,
        )

        self.assertEqual(scope, "truncated")
        self.assertEqual(note, None)
        self.assertIsNotNone(indices)
        assert indices is not None
        self.assertEqual(indices.tolist(), list(range(28)))

    def test_select_runs_prefers_eligible_run_over_too_short_candidate(self) -> None:
        from experiments.tep.exp02_fault_propagation_analysis.run import select_runs

        selected = select_runs(
            subset_runs=[
                {"run_key": "M4_d01", "scenario": "d01", "mode": "M4", "window_count": 10},
                {"run_key": "M1_d01", "scenario": "d01", "mode": "M1", "window_count": 7073},
            ],
            representative_cfg={
                "scenarios": ["d01"],
                "run_keys": [],
                "modes": [],
                "mode_preference": ["M4", "M1"],
                "max_runs_per_scenario": 1,
            },
            min_windows_to_process=16,
        )

        self.assertEqual([row["run_key"] for row in selected], ["M1_d01"])

    def test_external_config_load_stays_anchored_to_repo_root(self) -> None:
        from experiments.tep.exp02_fault_propagation_analysis.run import load_config, load_yaml

        default_config = load_yaml(
            ROOT / "experiments/tep/exp02_fault_propagation_analysis/config.yaml"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            external_config = Path(tmp_dir) / "external_config.yaml"
            external_config.write_text(yaml.safe_dump(default_config, sort_keys=False), encoding="utf-8")
            loaded = load_config(external_config)

        self.assertEqual(
            Path(loaded["paths"]["bundle_path"]),
            ROOT / "data/processed/tep/tep_processed_manifest.json",
        )
        self.assertEqual(
            Path(loaded["paths"]["processed_root"]),
            ROOT / "data/processed/tep",
        )

    def test_plot_prepare_clean_directory_removes_stale_png(self) -> None:
        from experiments.tep.exp02_fault_propagation_analysis.plot import prepare_clean_directory

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "figures"
            target.mkdir(parents=True, exist_ok=True)
            stale = target / "stale.png"
            stale.write_text("old", encoding="utf-8")

            prepare_clean_directory(target)

            self.assertTrue(target.exists())
            self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
