"""Cross-check raw/processed sample scale against literature-oriented references."""

from __future__ import annotations

from pathlib import Path
import unittest
import json

import numpy as np

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None

ROOT = Path(__file__).resolve().parents[2]


def _mat_main_matrix(path: Path) -> np.ndarray:
    payload = loadmat(path)  # type: ignore[arg-type]
    for value in payload.values():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim == 2 and value.shape[1] >= 53:
            return np.asarray(value)
    raise AssertionError(f"no suitable 2D matrix in {path}")


def _mat_vector(path: Path, *candidates: str) -> np.ndarray:
    payload = loadmat(path)  # type: ignore[arg-type]
    for name in candidates:
        if name in payload and isinstance(payload[name], np.ndarray):
            v = np.asarray(payload[name])
            if v.ndim == 1:
                return v
            if v.ndim == 2 and 1 in v.shape:
                return np.asarray(v).reshape(-1)
    # fallback first column-vector found
    for value in payload.values():
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 1:
            return np.asarray(value).reshape(-1)
    raise AssertionError(f"cannot find candidates {candidates} in {path}")


def _split_counts(manifest_path: Path) -> dict[str, int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    indices = payload.get("split_indices", {})
    if isinstance(indices, dict):
        return {k: int(len(v)) for k, v in indices.items()}
    # fallback for older summary style
    for key in ("split_summary", "counts", "sample_counts"):
        value = payload.get(key)
        if isinstance(value, dict):
            return {k: int(v) for k, v in value.items()}
    raise AssertionError(f"cannot infer split counts from {manifest_path}")


def _expected_nonlinear_processed_counts() -> dict[str, int]:
    raw_root = ROOT / "data" / "raw" / "nonlinear"
    nl = {}

    duffing_raw = len(_mat_vector(raw_root / "duffing" / "DATA_EMPS.mat", "qg", "u"))
    nl["duffing"] = duffing_raw - 128 - 1 + 1

    silver_raw = len(_mat_vector(raw_root / "silverbox" / "SNLS80mV.mat", "V1", "u"))
    nl["silverbox"] = silver_raw - 128 - 1 + 1

    mat = loadmat(raw_root / "cascaded_tanks" / "CascadedTanksFiles" / "dataBenchmark.mat")
    casc = int(np.asarray(mat["uEst"]).reshape(-1).shape[0])
    nl["cascaded_tanks"] = 2 * (casc - 128 - 1 + 1)

    vw = len(_mat_vector(raw_root / "volterra_wiener" / "WienerHammerBenchMark.mat", "uBenchMark", "u"))
    nl["volterra_wiener"] = vw - 128 - 1 + 1

    # 5 trajectories in coupled: one 500-sample uniform_1/2 and 3 PRBS trajectories.
    nl["coupled_duffing"] = 5 * (500 - 128 - 1 + 1)
    return nl


if loadmat is None:  # pragma: no cover
    @unittest.skip("scipy is required for raw `.mat` inspection")
    class DataScaleAuditTest(unittest.TestCase):
        pass

else:

    class DataScaleAuditTest(unittest.TestCase):
        def _assert_equal(self, got: int, expected: int, msg: str) -> None:
            self.assertEqual(
                got,
                expected,
                f"{msg} (got={got}, expected={expected})",
            )

        def test_nonlinear_raw_alignment_with_reference_protocol(self) -> None:
            raw_root = ROOT / "data" / "raw" / "nonlinear"

            # Nonlinear benchmark strict references supplied by the protocol:
            # Silverbox: 131072 I/O samples.
            silver = _mat_vector(raw_root / "silverbox" / "SNLS80mV.mat", "V1", "u")
            self._assert_equal(len(silver), 131072, "Silverbox raw sample count")

            # Cascaded Tanks: 1024 + 1024 estimation / validation records.
            casc = loadmat(raw_root / "cascaded_tanks" / "CascadedTanksFiles" / "dataBenchmark.mat")  # type: ignore[arg-type]
            self._assert_equal(int(np.asarray(casc["uEst"]).reshape(-1).shape[0]), 1024, "Cascaded Tanks uEst record length")
            self._assert_equal(int(np.asarray(casc["uVal"]).reshape(-1).shape[0]), 1024, "Cascaded Tanks uVal record length")

            # Coupled Electric Drives / DATAUNIF + DATAPRBS 500 samples per trajectory.
            uniform = loadmat(raw_root / "coupled_duffing" / "DATAUNIF.MAT")  # type: ignore[arg-type]
            prbs = loadmat(raw_root / "coupled_duffing" / "DATAPRBS.MAT")  # type: ignore[arg-type]
            for name in ("u11", "u12", "z11", "z12"):
                self._assert_equal(len(np.asarray(uniform[name]).reshape(-1)), 500, f"coupled/Uniform trajectory {name}")
            for name in ("u1", "z1", "u2", "z2", "u3", "z3"):
                self._assert_equal(len(np.asarray(prbs[name]).reshape(-1)), 500, f"coupled/PRBS trajectory {name}")

        def test_nonlinear_processed_split_scale_is_windowed_and_explainable(self) -> None:
            root = ROOT / "data" / "splits" / "nonlinear"
            expected = _expected_nonlinear_processed_counts()
            for dataset, expected_count in expected.items():
                manifest = ROOT / "data" / "splits" / "nonlinear" / f"{dataset}_split_manifest.json"
                counts = _split_counts(manifest)
                total = sum(counts.values())
                self.assertIn("train", counts)
                self.assertIn("val", counts)
                self.assertIn("test", counts)
                self._assert_equal(total, expected_count, f"{dataset} processed split total")

            # also ensure each split is disjoint by index in current split manifest files.
            for dataset in expected.keys():
                payload = json.loads((root / f"{dataset}_split_manifest.json").read_text(encoding="utf-8"))
                indices = payload.get("split_indices", {})
                train = set(indices.get("train", []))
                val = set(indices.get("val", []))
                test = set(indices.get("test", []))
                self.assertTrue(train.isdisjoint(val))
                self.assertTrue(train.isdisjoint(test))
                self.assertTrue(val.isdisjoint(test))

        def test_hydraulic_raw_cycle_scale(self) -> None:
            raw_root = ROOT / "data" / "raw" / "hydraulic"
            channels = [
                "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
                "FS1", "FS2", "TS1", "TS2", "TS3", "TS4",
                "VS1", "EPS1", "SE", "CP", "CE",
            ]

            rows = []
            for ch in channels:
                arr = np.loadtxt(raw_root / f"{ch}.txt")
                arr = np.asarray(arr)
                self.assertEqual(arr.ndim, 2)
                self.assertGreater(arr.shape[0], 0)
                self.assertGreater(arr.shape[1], 0)
                rows.append(arr.shape[0])

            self.assertEqual(sorted(set(rows)), [2205], "hydraulic channels must align to 2205 cycles")
            self.assertEqual(len(rows), 17)

            profile = np.loadtxt(raw_root / "profile.txt")
            profile = np.asarray(profile)
            self.assertEqual(profile.ndim, 2)
            self.assertEqual(profile.shape, (2205, 5))

        def test_tep_raw_layout_and_reference_run_counts(self) -> None:
            raw_root = ROOT / "data" / "raw" / "tep"
            modes = ["M1", "M2", "M3", "M4", "M5", "M6"]

            # protocol-level reference target for run-level organization
            # raw run length in local files is not fixed 500/960; this is checked by this test.
            for mode in modes:
                mode_root = raw_root / mode
                files = sorted([p for p in mode_root.glob("*.mat") if p.is_file()])
                self.assertEqual(len(files), 29, f"{mode} should have 29 runs")
                self.assertTrue(any(p.stem.lower() == f"{mode.lower()}d00" for p in files), f"{mode}: missing d00")

                lens = []
                for p in files:
                    mat = _mat_main_matrix(p)
                    self.assertEqual(mat.shape[1], 81)
                    lens.append(mat.shape[0])
                length_set = sorted(set(lens))
                self.assertGreaterEqual(len(length_set), 1)
                self.assertGreater(length_set[0], 0)
                self.assertGreater(max(lens), 500)
                # Paper-level fixed 500/960 reference values are not preserved at this raw run-file level.
                self.assertFalse(set(lens).issubset({500, 960}), f"{mode}: raw run lengths are reduced to fixed 500/960 unexpectedly")


if __name__ == "__main__":
    unittest.main()
