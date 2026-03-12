from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "experiments" / "nonlinear" / "exp03_ablation_hparam"
BASE_CONFIG = SCRIPT_DIR / "config.yaml"


def _write_test_config(target: Path) -> Path:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    payload["experiment"]["output_root"] = str((target / "results").resolve())
    for dataset_spec in payload["datasets"].values():
        dataset_spec["manifest"] = str((SCRIPT_DIR / dataset_spec["manifest"]).resolve())
    payload["suites"]["smoke"]["slice"] = {"train": 32, "val": 16, "test": 16}
    payload["suites"]["smoke"]["method_overrides"].update(
        {
            "epochs": 1,
            "batch_size": 16,
            "eval_batch_size": 16,
            "feature_chunk_size": 128,
            "max_canonical_terms_per_order": 10000,
            "max_full_recovery_elements": 100000,
        }
    )
    config_path = target / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return config_path


def _run_script(script_name: str, *args: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name), *args],
        cwd=ROOT,
        env=env,
        check=True,
    )


def test_exp03_run_collect_plot_smoke(tmp_path: Path) -> None:
    config_path = _write_test_config(tmp_path)

    _run_script("run.py", "--config", str(config_path), "--suite", "smoke", "--mode", "ablation", "--limit", "1")
    _run_script("run.py", "--config", str(config_path), "--suite", "smoke", "--mode", "hyperparameter", "--limit", "1")
    _run_script("collect.py", "--config", str(config_path), "--suite", "smoke")
    _run_script("plot.py", "--config", str(config_path), "--suite", "smoke")

    result_root = tmp_path / "results" / "smoke"
    assert next(result_root.glob("runs/ablation/*/result.json")).exists()
    assert next(result_root.glob("runs/hyperparameter/*/result.json")).exists()
    assert (result_root / "summaries" / "table_5_1_8.csv").exists()
    assert (result_root / "summaries" / "table_5_1_9.csv").exists()
    assert (result_root / "plots" / "ablation_comparison.png").exists()
    assert (result_root / "plots" / "hyperparameter_sensitivity.png").exists()
