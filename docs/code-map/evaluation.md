# Evaluation / Core Handoff 模块地图

目录：`code/evaluation/`

## 职责

Evaluation 层是 `a-share-alpha-lab` 与 `alpha-research-core` 的桥接层。它负责读取本项目 full factor table，按年份或日期在内存中过滤，构造 core `EvaluationSpec`，然后调用 `run_evaluation(spec)`。本项目不修改 core 项目文件。

## 关键文件

| 文件 | 作用 |
|---|---|
| `run_factor_evaluation.py` | 本项目 evaluation CLI 入口 |
| `core_bridge.py` | full factor 读取、列发现、`EvaluationSpec` 构造和 core 调用 |
| `build_factor_summary.py` | 从 core artifacts 汇总 `factor_summary.csv/md` |
| `compare_factor_summaries.py` | 诊断 full breakdown 与 yearly baseline 的差异 |

## Evaluation 主线

```text
run_factor_evaluation.py
-> parse CLI args
-> core_bridge.run_processed_factor_evaluation()
-> factor_full_path_covering_range()
-> discover factor / label columns
-> read_evaluation_frame()
-> make_a_share_evaluation_spec()
-> alpha_research_core.run_evaluation(spec)
-> build_factor_summary.py
```

## 输入合同

当前输入来源：

```text
data/processed_private/csmar/price_volume_factors_full/
  price_volume_factors_2014_2025.parquet
```

slim factor Parquet 必须保留：

| 类别 | 字段 |
|---|---|
| key | `stock_code`, `trade_date` |
| 过滤列 | `eligible`, `can_enter`, `can_exit` |
| 常用辅助列 | `market_type`, `amount`, `turnover`, `total_market_cap`, `float_market_cap` |
| factor | 例如 `mom_20d`, `vol_60d`, `amihud_120d` |
| label | 例如 `forward_return_20d` |
| PnL return | 默认 `forward_return_1d` |

slim factor Parquet 不保存完整 OHLC、复权价格、公司静态字段、行业详细字段。需要追溯时从 `panel_full/` 选择性读取后 join。

## 当前主口径

正式口径是：

```text
price_volume_factors_full/price_volume_factors_2014_2025.parquet
-> one full core run for 2024-2025
-> core metrics_by_year
```

不是：

```text
2024 independent run
+ 2025 independent run
```

Core 年度 breakdown 不清仓、不重跑组合。它从完整 full run 的 `long_short` path 切片，并在年度内重新计算 metrics。2025 年初如果出现从 2024 年底延续来的持仓 PnL，这是连续回测口径下的预期现象。

默认输出：

```text
reports/private_real/factor_evaluation/
  full_2024_2025/
    label_availability.csv
    evaluation_manifest.json
    factor_summary.csv
    factor_summary.md
    core_runs/
  comparisons/full_vs_yearly_current_2024_2025/
```

`run_factor_evaluation.py` 支持闭区间日期过滤。本项目读取 full slim factor Parquet 后在内存中截取，不创建 `core_inputs/`，也不生成 subperiod Parquet。若要做事件切片或 yearly independent diagnostic，需要显式指定独立 `--report-root`，避免覆盖正式 full 结果。

## CLI 到 core 参数映射

| CLI 参数 | core 字段 |
|---|---|
| `--labels` | `EvaluationSpec.label_columns` |
| `--factors` / `--all-factors` | `EvaluationSpec.feature_columns` |
| `--pnl-return-column` | `EvaluationSpec.pnl_return_column` |
| `--cost-bps` | `WeightConfig.cost_bps` |
| `--engine` | `EvaluationSpec.engine` |
| `--segment-columns` | `EvaluationSpec.segment_columns` |
| `--risk-column` | `WeightConfig.risk_column` |
| `--neutralization-method` | `NeutralizationConfig.method` |
| `--neutralization-group-columns` | `NeutralizationConfig.group_columns` |
| `--neutralization-exposure-columns` | `NeutralizationConfig.exposure_columns` |

## 函数阅读路线

| 顺序 | 函数 | 阅读目标 |
|---:|---|---|
| 1 | `run_factor_evaluation.main()` | 看懂 CLI 参数默认值和传递方式 |
| 2 | `runtime.paths.factor_full_path_covering_range()` | 看懂请求年份如何匹配 full factor 文件 |
| 3 | `core_bridge.discover_factor_columns()` / `discover_label_columns()` | 看懂列发现规则 |
| 4 | `core_bridge.read_evaluation_frame()` | 看懂传给 core 的 DataFrame 包含哪些列 |
| 5 | `core_bridge.make_a_share_evaluation_spec()` | 看懂本项目如何填充 core `EvaluationSpec` |
| 6 | `core_bridge.run_label_evaluation()` | 看懂每个 label 如何生成一个 core run |
| 7 | `build_factor_summary.build_factor_summary_frame()` | 看懂 core outputs 如何变成项目级 summary |
| 8 | `compare_factor_summaries.build_comparison_frame()` | 看懂 full vs diagnostic yearly 的比较口径 |

## 读源码时的断点建议

- `parquet_path`
- `feature_columns`
- `labels`
- `range_start` / `range_end`
- `evaluation_frame.columns`
- `spec.pnl_return_column`
- `spec.tradability_config`
- `spec.time_breakdowns`
- `output_dir`

## 常见误读

- `evaluation_manifest.json` 是本项目索引，不是 core 标准 `manifest.json`。
- `factor_summary.csv` 是轻量汇总，不是全部评价明细。
- `metrics_by_year` 是 core 的年度 breakdown，不是 yearly independent evaluation。
- 旧 yearly baseline 没有 `pnl_return_column="forward_return_1d"`，不能拿来判断当前 PnL 口径。
- 如果要做“每年空仓重启”，必须显式另跑 yearly independent evaluation，不能混入正式 full breakdown。
