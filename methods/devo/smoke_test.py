"""Minimal smoke test for DeVo."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from data.adapters.base import DatasetArtifacts, DatasetBundle, DatasetMeta, DatasetSplit, TaskFamily
from methods.utils import load_processed_dataset_bundle

from .model import DeVoConfig
from .trainer import DeVoMethod


def build_toy_bundle(seed: int = 7) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_train, n_val, n_test = 256, 64, 64
    window_length, input_dim, output_dim, horizon = 6, 2, 2, 1
    total = n_train + n_val + n_test

    X = rng.normal(size=(total, window_length, input_dim)).astype(np.float32)
    y0 = (
        0.7 * X[:, 0, 0]
        - 0.3 * X[:, 1, 1]
        + 0.5 * X[:, 2, 0] * X[:, 2, 1]
        + 0.25 * X[:, 3, 0] ** 2
    )
    y1 = (
        -0.2 * X[:, 0, 1]
        + 0.4 * X[:, 1, 0]
        + 0.35 * X[:, 4, 0] * X[:, 1, 1]
        + 0.30 * X[:, 5, 1] ** 2
    )
    Y = np.stack((y0, y1), axis=-1)[:, None, :].astype(np.float32)

    def split(start: int, stop: int) -> DatasetSplit:
        return DatasetSplit(
            X=X[start:stop],
            Y=Y[start:stop],
            sample_id=np.arange(start, stop),
            meta={},
        )

    return DatasetBundle(
        train=split(0, n_train),
        val=split(n_train, n_train + n_val),
        test=split(n_train + n_val, total),
        meta=DatasetMeta(
            dataset_name="toy_devo",
            task_family=TaskFamily.NONLINEAR,
            input_dim=input_dim,
            output_dim=output_dim,
            window_length=window_length,
            horizon=horizon,
            split_protocol="toy_holdout_v1",
            has_ground_truth_kernel=False,
            has_ground_truth_gfrf=False,
            extras={"task_usage": ["prediction", "kernel_recovery"]},
        ),
        artifacts=DatasetArtifacts(),
    )


def load_real_bundle_if_available() -> Optional[DatasetBundle]:
    manifest = Path("data/processed/nonlinear/duffing/duffing_processed_manifest.json")
    if not manifest.exists():
        return None
    return load_processed_dataset_bundle(manifest)


def run_smoke_test() -> None:
    real_bundle = load_real_bundle_if_available()
    if real_bundle is not None:
        print(
            "[DeVo smoke] loaded dataset bundle:",
            real_bundle.meta.dataset_name,
            tuple(np.asarray(real_bundle.train.X).shape),
            tuple(np.asarray(real_bundle.train.Y).shape),
        )

    bundle = build_toy_bundle()
    method = DeVoMethod(
        DeVoConfig(
            orders=(1, 2),
            num_branches=2,
            epochs=3,
            batch_size=32,
            eval_batch_size=64,
            learning_rate=5e-3,
            feature_chunk_size=256,
            max_canonical_terms_per_order=20_000,
            verbose=True,
            log_every=1,
        )
    )
    method.fit(bundle)

    pred = method.predict(bundle.test.X[:8])
    recovered = method.recover_kernels(materialize_full=True, max_full_elements=500_000)
    recovered_orders = recovered.kernels["orders"]
    attributions = method.attribute(bundle.test.X[:8], bundle.test.Y[:8])
    exported = method.export_parameters()

    assert method.supports_kernel_recovery() is True
    assert pred.shape == bundle.test.Y[:8].shape
    assert attributions.shape == bundle.test.X[:8].shape
    assert len(recovered_orders) == 2
    assert recovered_orders["1"]["canonical_indices"].shape[1] == 1
    assert recovered_orders["2"]["canonical_indices"].shape[1] == 2
    assert recovered_orders["2"]["full_tensor"] is not None
    assert exported["orders"][2]["effective_parameters"].shape[-1] == recovered_orders["2"]["feature_count"]

    print("[DeVo smoke] device:", method.device.type)
    print("[DeVo smoke] prediction shape:", pred.shape)
    print("[DeVo smoke] attribution shape:", attributions.shape)
    print(
        "[DeVo smoke] recovered orders:",
        [
            {
                "order": order["order"],
                "feature_count": order["feature_count"],
                "full_tensor_shape": order["full_tensor_shape"],
            }
            for order in recovered_orders.values()
        ],
    )


if __name__ == "__main__":
    run_smoke_test()
