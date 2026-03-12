# Nonlinear Benchmark Experiment 01

This experiment implements the paper's main nonlinear prediction benchmark only:

- datasets: `duffing`, `silverbox`, `volterra_wiener`, `coupled_duffing`, `cascaded_tanks`
- methods: `narmax`, `tt_volterra`, `cp_volterra`, `laguerre_volterra`, `mlp`, `lstm`, `devo`
- metrics: `NMSE` and `RMSE`

Scope restrictions for this experiment:

- `coupled_duffing` and `cascaded_tanks` are used only for prediction.
- `volterra_wiener` is evaluated only for prediction here. No kernel-recovery or `KNMSE` path is used.
- The experiment layer does not change methods core logic.

## Directory layout

- `run.py`: executes dataset x method x seed runs and writes one `result.json` per run
- `collect.py`: scans saved runs and produces csv / markdown / latex tables
- `plot.py`: reads collected summary files and generates comparison figures
- `config.yaml`: full and smoke profiles

Output structure under one profile directory:

```text
outputs/<profile>/
  resolved_config.json
  run_manifest.json
  runs/<dataset>/<method>/seed_000/result.json
  summary/
    benchmark_summary_long.csv
    benchmark_nmse_table.csv
    benchmark_rmse_table.csv
    benchmark_tables.md
    benchmark_tables.tex
  plots/
    nmse_by_dataset.png
    rmse_by_dataset.png
```

## Metric definition

For one run:

- `RMSE = sqrt(mean((y_pred - y_true)^2))`
- `NMSE = MSE / mean((y_true - mean(y_true))^2)`

If the test target variance is numerically zero, the implementation falls back to signal power normalization.

## Usage

Smoke:

```bash
python experiments/nonlinear/exp01_prediction_benchmark/run.py --profile smoke
python experiments/nonlinear/exp01_prediction_benchmark/collect.py --profile smoke
python experiments/nonlinear/exp01_prediction_benchmark/plot.py --profile smoke
```

Full benchmark:

```bash
python experiments/nonlinear/exp01_prediction_benchmark/run.py --profile full
python experiments/nonlinear/exp01_prediction_benchmark/collect.py --profile full
python experiments/nonlinear/exp01_prediction_benchmark/plot.py --profile full
```

Useful overrides:

```bash
python experiments/nonlinear/exp01_prediction_benchmark/run.py \
  --profile full \
  --datasets duffing silverbox \
  --methods mlp lstm devo \
  --seeds 0 1 \
  --output-dir experiments/nonlinear/exp01_prediction_benchmark/outputs/custom
```

## Notes

- Each run writes `result.json` before execution with status `running`, then overwrites it with `success`, `failed`, or `skipped`.
- Shape or unsupported-input failures are classified as `skipped` so the full sweep does not crash.
- `plot.py` never retrains. It only reads `summary/benchmark_summary_long.csv` emitted by `collect.py`.
