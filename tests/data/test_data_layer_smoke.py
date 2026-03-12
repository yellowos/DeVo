"""Adapter contract and data-layer metadata consistency smoke tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.adapters import HydraulicAdapter, NonlinearAdapter, TEPAdapter
from data.adapters.base import TaskFamily
from data.builders.build_hydraulic import run_build as run_hydraulic_build
from data.builders.build_tep import run_build as run_tep_build
from data.checks.nonlinear_metadata_checks import (
    validate_benchmark_manifest,
    validate_cross_manifest_consistency,
    validate_truth_manifest,
)
from data.checks.unified_data_layer_checks import CheckReport, validate_data_layer, validate_generic_bundle


def _build_split(
    rng: np.random.Generator,
    *,
    num_samples: int,
    window_length: int = 8,
    input_dim: int = 2,
    output_dim: int = 1,
) -> dict[str, np.ndarray]:
    x = rng.normal(size=(num_samples, window_length, input_dim)).astype(np.float32)
    y_core = 0.45 * x[:, 0, 0]
    if window_length > 1:
        y_core = y_core - 0.10 * x[:, 1, 0]
    if input_dim > 1:
        y_core = y_core + 0.20 * x[:, :, 1].mean(axis=1)
    y_core = y_core.astype(np.float32)
    y = np.repeat(y_core[:, None, None], repeats=output_dim, axis=2)
    return {
        "X": x,
        "Y": y,
        "sample_id": np.arange(num_samples, dtype=np.int32),
    }


class DataAdapterSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[2]

    def test_nonlinear_adapter_enriches_metadata_and_artifacts(self) -> None:
        rng = np.random.default_rng(13)
        bundle = NonlinearAdapter.build_bundle(
            dataset_name="duffing",
            train=_build_split(rng, num_samples=16, input_dim=1),
            val=_build_split(rng, num_samples=8, input_dim=1),
            test=_build_split(rng, num_samples=8, input_dim=1),
            meta={"extras": {"unit": "smoke"}},
        )

        self.assertEqual(bundle.meta.task_family, TaskFamily.NONLINEAR)
        self.assertEqual(bundle.meta.dataset_name, "duffing")
        self.assertTrue(bundle.meta.has_ground_truth_kernel)
        self.assertTrue(bundle.meta.has_ground_truth_gfrf)
        self.assertEqual(bundle.meta.input_dim, 1)
        self.assertEqual(bundle.meta.output_dim, 1)
        self.assertIn("nonlinear_temporal_grouped_holdout_v1", bundle.meta.split_protocol)
        self.assertTrue(bundle.artifacts.protocol_file is not None)
        self.assertTrue(str(bundle.artifacts.protocol_file).endswith(".json"))
        self.assertIsNotNone(bundle.artifacts.grouping_file)

    def test_hydraulic_and_tep_adapters_keep_bundle_contract(self) -> None:
        rng = np.random.default_rng(17)
        base_meta = {
            "dataset_name": "toy_hydraulic",
            "task_family": "hydraulic",
            "input_dim": 3,
            "output_dim": 4,
            "window_length": 6,
            "horizon": 1,
            "split_protocol": "unit_test_protocol",
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
        }

        hydraulic_bundle = HydraulicAdapter.build_bundle(
            dataset_name="toy_hydraulic_unit",
            train=_build_split(rng, num_samples=12, window_length=6, input_dim=3, output_dim=4),
            val=_build_split(rng, num_samples=4, window_length=6, input_dim=3, output_dim=4),
            test=_build_split(rng, num_samples=6, window_length=6, input_dim=3, output_dim=4),
            meta=base_meta,
        )
        self.assertEqual(hydraulic_bundle.meta.task_family, TaskFamily.HYDRAULIC)
        self.assertEqual(hydraulic_bundle.meta.dataset_name, "toy_hydraulic")

        tep_meta = {
            **base_meta,
            "dataset_name": "tep_toy",
            "task_family": "tep",
        }
        tep_bundle = TEPAdapter.build_bundle(
            dataset_name="tep_toy",
            train=_build_split(rng, num_samples=10, window_length=6, input_dim=3, output_dim=4),
            val=_build_split(rng, num_samples=5, window_length=6, input_dim=3, output_dim=4),
            test=_build_split(rng, num_samples=5, window_length=6, input_dim=3, output_dim=4),
            meta=tep_meta,
        )
        self.assertEqual(tep_bundle.meta.task_family, TaskFamily.TEP)
        self.assertEqual(tep_bundle.meta.dataset_name, "tep_toy")

    def test_manifest_consistency_checks_succeed_for_non_linear_reference_files(self) -> None:
        metadata_root = self.root / "data" / "metadata" / "nonlinear"
        benchmark_payload = json.loads((metadata_root / "benchmark_manifest.json").read_text(encoding="utf-8"))
        kernel_payload = json.loads((metadata_root / "kernel_truth_manifest.json").read_text(encoding="utf-8"))
        gfrf_payload = json.loads((metadata_root / "gfrf_truth_manifest.json").read_text(encoding="utf-8"))

        validate_benchmark_manifest(benchmark_payload)
        validate_truth_manifest(kernel_payload, "kernel")
        validate_truth_manifest(gfrf_payload, "gfrf")
        validate_cross_manifest_consistency(
            benchmark_payload,
            kernel_payload,
            gfrf_payload,
        )

    def test_split_leakage_is_detected(self) -> None:
        rng = np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "data"
            processed_root = data_root / "processed" / "nonlinear" / "toy_leak"
            split_root = data_root / "splits" / "nonlinear"
            processed_root.mkdir(parents=True, exist_ok=True)
            split_root.mkdir(parents=True, exist_ok=True)

            def write_split(name: str, num_samples: int, split_indices: np.ndarray) -> None:
                split = _build_split(rng, num_samples=num_samples, window_length=4, input_dim=2, output_dim=1)
                np.save(processed_root / f"{name}_X.npy", split["X"])
                np.save(processed_root / f"{name}_Y.npy", split["Y"])
                np.save(processed_root / f"{name}_sample_id.npy", split_indices)

            train_indices = np.array([0, 1, 2, 3], dtype=np.int32)
            val_indices = np.array([2, 3, 4], dtype=np.int32)  # overlap with train
            test_indices = np.array([5, 6], dtype=np.int32)
            write_split("train", len(train_indices), train_indices)
            write_split("val", len(val_indices), val_indices)
            write_split("test", len(test_indices), test_indices)

            split_path = split_root / "toy_leak_split_manifest.json"
            split_payload = {
                "split_indices": {
                    "train": train_indices.tolist(),
                    "val": val_indices.tolist(),
                    "test": test_indices.tolist(),
                }
            }
            split_path.write_text(json.dumps(split_payload), encoding="utf-8")

            manifest = {
                "bundle_meta": {
                    "dataset_name": "toy_leak",
                    "task_family": "nonlinear",
                    "input_dim": 2,
                    "output_dim": 1,
                    "window_length": 4,
                    "horizon": 1,
                    "split_protocol": "nonlinear_temporal_grouped_holdout_v1",
                    "has_ground_truth_kernel": False,
                    "has_ground_truth_gfrf": False,
                    "extras": {},
                },
                "bundle_artifacts": {},
                "processed_files": {
                    "train_X": "train_X.npy",
                    "train_Y": "train_Y.npy",
                    "train_sample_id": "train_sample_id.npy",
                    "val_X": "val_X.npy",
                    "val_Y": "val_Y.npy",
                    "val_sample_id": "val_sample_id.npy",
                    "test_X": "test_X.npy",
                    "test_Y": "test_Y.npy",
                    "test_sample_id": "test_sample_id.npy",
                },
                "split_file": "splits/nonlinear/toy_leak_split_manifest.json",
            }
            manifest_path = processed_root / "toy_leak_processed_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = CheckReport()
            ok = validate_generic_bundle(
                report=report,
                dataset="toy_leak",
                data_root=data_root,
                family=TaskFamily.NONLINEAR,
                processed_root=processed_root,
                split_root=split_root,
            )

            self.assertFalse(ok)
            self.assertTrue(any("leakage" in entry.lower() for entry in report.errors))

    def test_silverbox_split_manifest_is_leakage_free(self) -> None:
        split_payload = json.loads(
            (self.root / "data" / "splits" / "nonlinear" / "silverbox_split_manifest.json").read_text(encoding="utf-8")
        )
        indices = split_payload.get("split_indices", {})
        train = set(indices.get("train", []))
        val = set(indices.get("val", []))
        test = set(indices.get("test", []))
        self.assertIsInstance(train, set)
        self.assertIsInstance(val, set)
        self.assertIsInstance(test, set)
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(val.isdisjoint(test))

        manifest_path = (
            self.root
            / "data"
            / "processed"
            / "nonlinear"
            / "silverbox"
            / "silverbox_processed_manifest.json"
        )
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("processed_files", {})
        train_sample_id = np.load(self.root / "data" / "processed" / "nonlinear" / "silverbox" / files["train_sample_id"], allow_pickle=True)
        val_sample_id = np.load(self.root / "data" / "processed" / "nonlinear" / "silverbox" / files["val_sample_id"], allow_pickle=True)
        test_sample_id = np.load(self.root / "data" / "processed" / "nonlinear" / "silverbox" / files["test_sample_id"], allow_pickle=True)
        self.assertTrue(set(train_sample_id.tolist()).isdisjoint(set(val_sample_id.tolist())))
        self.assertTrue(set(train_sample_id.tolist()).isdisjoint(set(test_sample_id.tolist())))
        self.assertTrue(set(val_sample_id.tolist()).isdisjoint(set(test_sample_id.tolist())))

    def test_validate_generic_bundle_falls_back_from_missing_absolute_split_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "data"
            processed_root = data_root / "processed" / "nonlinear" / "toy_portable"
            split_root = data_root / "splits" / "nonlinear"
            processed_root.mkdir(parents=True, exist_ok=True)
            split_root.mkdir(parents=True, exist_ok=True)

            split_indices: dict[str, list[int]] = {}
            split_layout = {"train": 4, "val": 2, "test": 2}
            split_offsets = {"train": 0, "val": 100, "test": 200}
            for split_name, num_samples in split_layout.items():
                split = _build_split(
                    np.random.default_rng(23 + len(split_name)),
                    num_samples=num_samples,
                    window_length=4,
                    input_dim=2,
                    output_dim=1,
                )
                sample_ids = np.arange(
                    split_offsets[split_name],
                    split_offsets[split_name] + num_samples,
                    dtype=np.int32,
                )
                np.save(processed_root / f"{split_name}_X.npy", split["X"])
                np.save(processed_root / f"{split_name}_Y.npy", split["Y"])
                np.save(processed_root / f"{split_name}_sample_id.npy", sample_ids)
                split_indices[split_name] = sample_ids.tolist()

            split_path = split_root / "toy_portable_split_manifest.json"
            split_path.write_text(json.dumps({"split_indices": split_indices}), encoding="utf-8")

            manifest = {
                "bundle_meta": {
                    "dataset_name": "toy_portable",
                    "task_family": "nonlinear",
                    "input_dim": 2,
                    "output_dim": 1,
                    "window_length": 4,
                    "horizon": 1,
                    "split_protocol": "nonlinear_temporal_grouped_holdout_v1",
                    "has_ground_truth_kernel": False,
                    "has_ground_truth_gfrf": False,
                    "extras": {},
                },
                "bundle_artifacts": {},
                "processed_files": {
                    "train_X": "train_X.npy",
                    "train_Y": "train_Y.npy",
                    "train_sample_id": "train_sample_id.npy",
                    "val_X": "val_X.npy",
                    "val_Y": "val_Y.npy",
                    "val_sample_id": "val_sample_id.npy",
                    "test_X": "test_X.npy",
                    "test_Y": "test_Y.npy",
                    "test_sample_id": "test_sample_id.npy",
                },
                "split_file": "/tmp/nonexistent/toy_portable_split_manifest.json",
            }
            manifest_path = processed_root / "toy_portable_processed_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = CheckReport()
            ok = validate_generic_bundle(
                report=report,
                dataset="toy_portable",
                data_root=data_root,
                family=TaskFamily.NONLINEAR,
                processed_root=processed_root,
                split_root=split_root,
            )

            self.assertTrue(ok)
            self.assertEqual(report.errors, [])

    def test_hydraulic_raw_channel_layout_matches_protocol_expectation(self) -> None:
        raw_root = self.root / "data" / "raw" / "hydraulic"
        expected_channels = [
            "PS1",
            "PS2",
            "PS3",
            "PS4",
            "PS5",
            "PS6",
            "FS1",
            "FS2",
            "TS1",
            "TS2",
            "TS3",
            "TS4",
            "VS1",
            "EPS1",
            "SE",
            "CP",
            "CE",
        ]
        for channel in expected_channels:
            self.assertTrue((raw_root / f"{channel}.txt").exists(), msg=f"missing hydraulic channel file {channel}.txt")

        row_counts = []
        for channel in expected_channels:
            arr = np.loadtxt(raw_root / f"{channel}.txt")
            arr = np.atleast_2d(arr)
            self.assertEqual(arr.ndim, 2)
            self.assertGreater(arr.shape[1], 0)
            row_counts.append(arr.shape[0])

        self.assertEqual(len(set(row_counts)), 1, "all hydraulic channels must share one aligned cycle length")

        profile = np.loadtxt(raw_root / "profile.txt")
        profile = np.atleast_2d(profile)
        self.assertEqual(profile.ndim, 2)
        self.assertEqual(profile.shape[1], 5)
        self.assertEqual(profile.shape[0], row_counts[0])

    def test_tep_raw_layout_and_mode_file_naming(self) -> None:
        raw_root = self.root / "data" / "raw" / "tep"
        modes = ["M1", "M2", "M3", "M4", "M5", "M6"]
        for mode in modes:
            mode_root = raw_root / mode
            self.assertTrue(mode_root.exists(), msg=f"missing mode folder {mode}")
            files = sorted([p.name for p in mode_root.glob("m*.mat")])
            self.assertEqual(len(files), 29)
            self.assertEqual(files[0], f"{mode.lower()}d00.mat")
            self.assertEqual(files[-1], f"{mode.lower()}d28.mat")

        try:
            from scipy.io import loadmat
        except Exception:
            self.skipTest("scipy is required for TEP raw layout check")

        for mode in modes:
            mode_root = raw_root / mode
            sample_path = mode_root / f"{mode.lower()}d01.mat"
            data = loadmat(sample_path)
            first_matrix = None
            for value in data.values():
                if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] >= 81:
                    first_matrix = value
                    break
            self.assertIsNotNone(first_matrix)
            assert first_matrix is not None
            self.assertEqual(first_matrix.shape[1], 81)

    def test_unified_check_expectation_for_nonready_families(self) -> None:
        run_hydraulic_build(self.root)
        tep_manifest = self.root / "data" / "processed" / "tep" / "tep_processed_manifest.json"
        if not tep_manifest.exists():
            run_tep_build(self.root)
        report = CheckReport()
        report = validate_data_layer(project_root=self.root, task_family="all")
        self.assertEqual(report.errors, [])


if __name__ == "__main__":
    unittest.main()
