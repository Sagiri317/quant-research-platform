# Quant Research Platform 架构说明

`quant-research-platform` 是本地多市场量化研究平台。上游 market module 负责市场数据、信号、label、预测和研究语境；evaluation layer 负责从 long-format panel 到 schema `0.5` artifacts 的统一评价流程；API 和 React workbench 只消费 artifacts 与 `manifest.display`。

## 职责边界

平台 evaluation layer 负责：

- `EvaluationSpec`、`WeightConfig`、`ExecutionConfig`、`TradabilityConfig`、`CostConfig`、`NeutralizationConfig`。
- long panel schema validation。
- IC / RankIC、grouped returns、spread diagnostics。
- weights、positions、turnover、portfolio returns、cost、PnL 和 performance summary。
- `label_columns` 与 `pnl_return_column` 的口径拆分。
- optional neutralization、segment metrics、time breakdowns。
- `manifest.json`、`config.json`、`summary.json`、`combinations.csv`、CSV tables 和 `report.md`。
- completed evaluation-set artifacts 的读取、契约校验、comparison summary。
- 本地 FastAPI API 和 React/Vite evidence workbench。
- 本地 CSV / Parquet input helpers 和 `evaluate-parquet` CLI。

Market module 负责：

- 生成 long-format panel。
- 构造 feature、label、prediction 和 PnL return。
- 管理真实数据审批、路径、许可、Git ignore 和隐私边界。
- 选择 weight / cost / tradability / neutralization 参数。
- 解释市场语境和研究结论。

A 股模块额外负责：

- CSMAR 字段和许可边界。
- universe、eligible、can_enter、can_exit。
- 停牌、涨跌停、ST、上市天数等 domain rules。
- 因子、label、prediction、risk inputs 和 pipeline snapshot。

## 调用关系

```text
local approved data
-> market module
-> DataFrame / Parquet input
-> EvaluationSpec
-> run_evaluation
-> schema 0.5 evaluation-set artifacts
-> FastAPI evidence API
-> React evidence workbench
-> reports and research evidence
```

## A 股流程接口

A 股是当前唯一 enabled market module。它的真实私有数据流程入口分三层：

1. `quant_research_platform.markets.implementations.a_share.datasets.build_panel`：从 CSMAR interim yearly CSV 生成 `research_panel_<start_year>_<end_year>.parquet`。
2. `quant_research_platform.markets.implementations.a_share.factors.build_factors`：从 full panel 生成 `price_volume_factors_<start_year>_<end_year>.parquet`。
3. `quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation`：读取覆盖年份区间的 slim factor Parquet，并可用 `--start-date/--end-date` 对 `trade_date` 做评估阶段过滤，再通过 `EvaluationSpec` 调用 evaluation layer。

`evaluate-parquet` 属于通用 generic evaluation adapter，不保留 A 股 adapter 的 label availability summary、覆盖文件选择和私有 pipeline metadata。验证 A 股全流程或生成 A 股私有 runs 时，应优先使用 A 股专用评估入口。

前端 `Pipeline Builder` 默认写本地配置快照和 checkpoint。后端已支持 `execution_mode=processed_existing` 的轻量 runner，可以在已有 processed factor table 上调用 A 股 evaluation adapter；这仍不等同于完整 raw/interim/panel/factors build pipeline。后续要把“写因子表达式和配置后一键全流程”做完整，需要把该页面接到完整 A 股 build/evaluate 编排。

Factor Factory 当前已有后端 registry、批量评价、API 轻量入口和本地 workbench 浏览入口。P4/P5 小样本示例通过 `scripts/phase1_acceptance.py` 复跑 A 股 processed-existing acceptance runs、生成 Factor Factory candidate summary，并输出 old/new 比较报告。

## 源码布局

当前 Phase 1 结构不创建 `apps/api`，也不保留空 `markets/crypto/` 实现。

```text
src/quant_research_platform/
  __init__.py
  evaluations/          contracts、panel validation、evaluation_runner、results、breakdowns
  kernels/              dataframe reference path and tensor helpers
  artifacts/            writer、loader、display contract、inventory、comparison
  datasets/             generic CSV / Parquet readers
  interfaces/
    api/                local FastAPI evidence platform
    cli/                qrp command entry
  markets/
    implementations/
      a_share/          CSMAR-backed A-share market module
  pipelines/
    a_share/            A-share workflow runner and checkpoints
  research/             Factor Factory and future research contracts
  runtime/              shared runtime helpers

web/                    React/Vite local workbench
config/a_share/         factor templates and pipeline configs
```

## 输出契约

一个 evaluation set 写出：

```text
manifest.json
config.json
summary.json
combinations.csv
report.md
metrics/ic.csv
metrics/ic_summary.csv
metrics/grouped_returns.csv
metrics/group_spread_summary.csv
metrics/performance_summary.csv
portfolio/positions.csv
portfolio/turnover.csv
portfolio/returns.csv
portfolio/pnl.csv
portfolio/cost_breakdown.csv
portfolio/exposure_summary.csv
diagnostics/weight_diagnostics.csv
diagnostics/neutralization_diagnostics.csv
metrics_by_year/       # optional
metrics_by_quarter/    # optional
metrics_by_month/      # optional
```

所有明细 CSV 都显式带上：

```text
combination_id, feature, label, weight_id, cost_id
```

`manifest.json` 和 `summary.json` 均包含 `artifact_schema_version: "0.5"` 与 `artifact_type: "evaluation_set"`。Manifest 中的 artifact 路径必须是 portable relative paths，不写入 Windows drive letter、用户目录或绝对路径。

`manifest.display` 是本地 API、下载和 React workbench 的展示契约，集中声明可浏览表格、可下载条目、下载 preset 和允许渲染的图表配置。新增 artifact 时，writer / loader、API payload/download、frontend display consumers、tests 和 docs 必须同步。

## API 和平台

FastAPI layer 聚合本地 run roots，提供 run list、evidence payload、CSV preview、direct artifact download、ZIP download 和 comparison。API 只暴露当前 evaluation-set artifacts 和动态 comparison，不通过 endpoint 暴露未进入 `manifest.display` 的隐式文件。

API-visible run identity is `(source_alias, run_id)`. Run list、evidence routes、CSV preview、direct downloads、ZIP cache、comparison 和 React state 都必须保留 `source_alias`，不能只按 `run_id` 折叠。

Phase 1 中 `interfaces/api` 是本地 FastAPI 适配层；Phase 2 的 `apps/api` 是未来可部署 HTTP 应用装配层。两个 API 目录职责不同，不应合并，也不应在 Phase 1 提前创建 `apps/api`。

平台边界：

- API 是本地研究辅助，不是部署服务。
- `POST /api/runs` 只从用户提供的本地输入构造 `EvaluationSpec`。
- 删除接口只能按项目规则处理明确允许删除的本地 uploaded artifacts。
- React frontend 只消费 API 返回的数据模型，不直接读本地文件，不重新计算 IC、RankIC、positions 或 turnover。
- Factor/Pipeline Studio 只发起受控 pipeline 请求，不连接外部数据源或交易接口。
- Factor Factory 前端浏览只展示 API / artifact 结果，不在前端重新计算 IC、PnL、positions 或 turnover。
- 已完成 artifact ZIP packaging 下沉到 artifacts；后续仍需单独处理的架构债包括 frontend feature split、完整 A 股 build runner 以及 eligibility / tradability 诊断可见化。

## Comparison

Comparison table 只提供独立指标排名，不生成综合评分：

- `rank_mean_ic`、`rank_mean_rankic`、`rank_mean_final_return`：数值越高排名越靠前。
- `rank_mean_turnover`、`rank_total_cost`：数值越低排名越靠前。
- 不提供 `overall_score`，避免把不同研究目标混成单一主观评价口径。

上游调用方接入契约见 [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md)。
