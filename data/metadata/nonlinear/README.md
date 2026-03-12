# Nonlinear benchmark metadata

本目录定义 Nonlinear benchmark 的统一元数据协议。

- `benchmark_manifest.json`
  - 给出每个 benchmark 的统一描述
  - 字段至少包括：`benchmark_name`、`system_type`、`task_usage`、
    `input_channels`、`output_channels`、`default_window_length`、
    `default_horizon`、`has_ground_truth_kernel`、`has_ground_truth_gfrf`、
    `recommended_split_protocol`
  - `artifacts` 下给出真值/分组等对象引用（引用字符串）；
    具体路径或对象绑定由 `NonlinearAdapter` 装配到 `artifacts` 中。

- `kernel_truth_manifest.json`
  - 指明哪些 benchmark 有 ground-truth kernel。
  - 字段 `has_ground_truth_kernel` 与 `kernel_reference` 用于统一登记。

- `gfrf_truth_manifest.json`
  - 指明哪些 benchmark 有 GFRF 真值，或可由 kernel 派生。
  - 字段 `has_ground_truth_gfrf` 与 `gfrf_reference` 用于统一登记。

- `protocols/nonlinear_temporal_grouped_holdout_v1.json`
  - 标准时间切分协议示例。

`NonlinearAdapter` 会在装配 bundle 时：
1. 校验 dataset_name 是否在 benchmark manifest 注册。
2. 从三份 manifest 合并并补齐 meta。
3. 自动生成标准 `DatasetBundle` 的 `meta` 与 `artifacts`。
4. 保留 `meta.extras` 与 `artifacts.extra` 记录与真值/任务协议相关的扩展字段。
