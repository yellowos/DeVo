"""Minimal smoke test for the CP-Volterra baseline."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from methods.baselines.cp_volterra import CPVolterraMethod


def _make_synthetic_bundle(seed: int = 11) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    window_length = 8
    input_dim = 1
    horizon = 1
    output_dim = 1

    def draw_split(num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(num_samples, window_length, input_dim)).astype(np.float32)
        linear_factor = np.asarray([0.75, -0.5, 0.2, 0.1, 0.0, -0.15, 0.05, 0.1], dtype=np.float32)
        quad_factor_a = np.asarray([0.4, -0.3, 0.0, 0.2, 0.1, 0.0, 0.15, -0.1], dtype=np.float32)
        quad_factor_b = np.asarray([-0.2, 0.35, 0.1, 0.0, -0.1, 0.1, 0.2, 0.05], dtype=np.float32)
        flat = x.reshape(num_samples, window_length)
        linear_term = flat @ linear_factor
        quadratic_term = 0.45 * (flat @ quad_factor_a) * (flat @ quad_factor_b)
        y = (0.1 + linear_term + quadratic_term).reshape(num_samples, horizon, output_dim)
        y = y + 0.01 * rng.normal(size=y.shape).astype(np.float32)
        return x, y.astype(np.float32)

    train_x, train_y = draw_split(256)
    val_x, val_y = draw_split(64)
    test_x, test_y = draw_split(64)

    return {
        "train": {"X": train_x, "Y": train_y},
        "val": {"X": val_x, "Y": val_y},
        "test": {"X": test_x, "Y": test_y},
        "meta": {
            "dataset_name": "synthetic_cp_volterra",
            "task_family": "nonlinear",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "window_length": window_length,
            "horizon": horizon,
            "split_protocol": "synthetic_holdout_v1",
            "has_ground_truth_kernel": True,
            "has_ground_truth_gfrf": False,
            "extras": {},
        },
        "artifacts": {},
    }


def run_smoke_test() -> dict[str, float]:
    bundle = _make_synthetic_bundle()
    method = CPVolterraMethod(
        config={
            "max_order": 2,
            "order_ranks": {1: 2, 2: 2},
            "learning_rate": 0.03,
            "batch_size": 64,
            "num_epochs": 80,
            "patience": 12,
            "seed": 17,
        },
        device="cpu",
    )
    method.fit(bundle)
    predictions = method.predict(bundle["test"]["X"])
    assert predictions.shape == bundle["test"]["Y"].shape

    mse = float(np.mean((predictions - bundle["test"]["Y"]) ** 2))
    assert mse < 0.05, f"Smoke test MSE too high: {mse}"

    recovered = method.recover_kernels(expand_full=True, max_full_kernel_elements=50_000)
    assert recovered.kernels["orders"][0]["input_factors_raw"]
    assert recovered.kernels["orders"][0]["materialized_full_kernel"] is True

    with TemporaryDirectory() as tmpdir:
        export_root = Path(tmpdir)
        state_path = method.save(export_root / "state.json")
        restored = CPVolterraMethod.load(state_path)
        restored_predictions = restored.predict(bundle["test"]["X"])
        assert np.allclose(predictions, restored_predictions, atol=1e-5)
        artifacts = restored.export_artifacts(export_root / "artifacts")
        assert "cp_parameters" in artifacts

    return {"test_mse": mse}


if __name__ == "__main__":
    metrics = run_smoke_test()
    print(metrics)
