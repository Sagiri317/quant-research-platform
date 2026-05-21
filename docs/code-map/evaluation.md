# Evaluation / Core 模块地图

主要目录：

```text
src/quant_research_platform/core/
src/quant_research_platform/artifacts/
src/quant_research_platform/api/
src/quant_research_platform/markets/a_share/evaluation/
```

## 职责

Evaluation 层把 market module 生成的 factor / prediction panel 交给平台 core。A 股模块负责读取本项目 full factor table、按年份或日期过滤、构造 `EvaluationSpec`；core 负责执行 `run_evaluation(spec)` 并写出 schema `0.5` artifacts。API 和前端只消费 artifacts 与 `manifest.display`。

## 关键文件

| 文件或目录 | 作用 |
|---|---|
| `src/quant_research_platform/core/` | `EvaluationSpec`、panel validation、evaluation orchestration |
| `src/quant_research_platform/artifacts/` | writer、loader、display contract、inventory、comparison |
| `src/quant_research_platform/api/` | local evidence API、CSV preview、downloads、run repository |
| `src/quant_research_platform/markets/a_share/evaluation/` | A 股 full factor table -> `EvaluationSpec` bridge |
| `web/` | React/Vite evidence workbench |

## Evaluation 主线

```text
A-share factor table
-> market evaluation bridge
-> EvaluationSpec
-> run_evaluation(spec)
-> schema 0.5 artifacts
-> manifest.display
-> API evidence / CSV preview / downloads
-> React workbench
```

## 输入合同

当前 A 股输入来源：

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
-> one full core run for selected date range
-> optional time breakdowns
```

不是：

```text
yearly independent run
+ yearly independent run
```

年度 breakdown 不清仓、不重跑组合。它从完整 full run 的 portfolio return path 切片，并在年度内重新计算 metrics。A 股 long-only 主报告需要显式使用 `portfolio_mode=long_only`。

## 顶层 `evaluate-parquet` CLI 到 core 参数映射

| CLI 参数 | core 字段 |
|---|---|
| `--label` | `EvaluationSpec.label_columns` |
| `--feature` | `EvaluationSpec.feature_columns` |
| `--pnl-return-column` | `EvaluationSpec.pnl_return_column` |
| `--portfolio-mode` | `WeightConfig.portfolio_mode` |
| `--cost-bps` | cost config / weight config |
| `--engine` | `EvaluationSpec.engine` |
| `--segment-column` | `EvaluationSpec.segment_columns` |
| `--risk-column` | risk-scaled weighting input |

A 股专用脚本 `quant_research_platform.markets.a_share.evaluation.run_factor_evaluation`
仍使用复数参数 `--labels`、`--factors`、`--all-factors`、`--segment-columns`。
顶层 `qrp a-share evaluate` 当前是统一 CLI 下的帮助入口，正式 A 股端到端
pipeline 命令仍在后续阶段补齐。

## Artifact / API / Frontend 断点

排查 contract 问题时优先看：

- `artifact_schema_version` 是否为 `0.5`。
- `manifest.display.schema_version` 是否存在并被 loader / API 校验。
- 每个可浏览 CSV 是否在 `manifest.display.tables`。
- 每个下载项是否在 `manifest.display.downloads.items`。
- ZIP “全选”是否使用 display preset，而不是扫描目录。
- run identity 是否始终保留 `(source_alias, run_id)`，不能只按 `run_id` 折叠。
- run switching 和 combination switching 是否更新 summary、charts、CSV preview、downloads 和 report mode。

## 常见误读

- project-level summary manifest 不是 core 标准 `manifest.json`。
- `factor_summary.csv` 是轻量汇总，不是全部评价明细。
- `metrics_by_year` 是 core 的年度 breakdown，不是 yearly independent evaluation。
- 如果要做“每年空仓重启”，必须显式另跑 yearly independent evaluation，不能混入正式 full breakdown。
