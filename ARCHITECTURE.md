# Quant Research Platform 架构说明

`quant-research-platform` 是本地多市场量化研究平台。Market module 负责市场数据、信号、label、预测和研究语境；evaluation layer 负责把 long-format panel 或 market adapter 输出转换为 schema `0.6` artifacts；API 和 React workbench 只消费 artifacts 中的 `manifest.display`。

## 职责边界

Evaluation layer 负责：

- `EvaluationSpec`、`WeightConfig`、`ExecutionConfig`、`TradabilityConfig`、`CostConfig`、`NeutralizationConfig`。
- Long panel schema validation。
- IC / RankIC、grouped returns、spread diagnostics。
- Weights、turnover、portfolio returns、cost、cumulative returns 和 performance summary。
- `manifest.json`、`config.json`、`summary.json`、`dimensions/combinations.csv`、CSV tables 和 `report.md`。
- Completed evaluation-set artifacts 的读取、契约校验、comparison summary。
- 本地 FastAPI API 和 React/Vite evidence workbench。

Market module 负责：

- 生成 long-format panel。
- 构造 factor、label、prediction 和 return source。
- 管理真实数据审批、路径、许可、Git ignore 和隐私边界。
- 解释市场语境和研究结论。

A 股模块额外负责 CSMAR 字段、universe、eligible/can_enter/can_exit、停牌、涨跌停、ST、上市天数等 domain rules。

## 主链路

```text
local approved data
-> A-share market module
-> interim data / research panel / factor table
-> PipelineConfig
-> start_from runner
-> A-share evaluation adapter
-> EvaluationSpec
-> run_evaluation
-> schema 0.6 evaluation-set artifacts
-> manifest.display
-> FastAPI evidence API
-> React evidence workbench
```

## A 股 Pipeline 接口

Pipeline Builder 的前端默认写配置快照：

```http
POST /api/pipeline-snapshots
```

前端也可切换到执行模式；后端执行入口为：

```http
POST /api/pipeline-runs
```

公开执行合同包含 `kernel` 和 `start_from`，没有独立结束层。每次执行都会走到 evaluation：

```text
existing_factors -> evaluation
rebuild_factors  -> build_factors -> evaluation
rebuild_panel    -> build_panel -> build_factors -> evaluation
rebuild_interim  -> build_interim -> build_panel -> build_factors -> evaluation
```

统一 runner 位于 `src/quant_research_platform/pipelines/a_share/runner.py`，会写 `run_config.yaml`、`warnings.json`、`summary.md`、`checkpoints.json` 和 `runtime_summary.json`。失败信息使用稳定 failure code，避免暴露 private path。

Snapshot 状态和执行状态分别由对应资源读取：

```http
GET /api/pipeline-snapshots/{run_id}
GET /api/pipeline-snapshots/{run_id}/checkpoints
GET /api/pipeline-runs/{run_id}
GET /api/pipeline-runs/{run_id}/checkpoints
```

## A 股专用数据入口

A 股真实私有数据 runs 必须优先使用 A 股专用 adapter：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation --help
```

`evaluate-parquet` 属于通用 local adapter check，不保留 A 股 adapter 的 label availability summary、覆盖文件选择和私有 pipeline metadata。验证 A 股全流程或生成 A 股私有 runs 时，应优先使用 A 股专用入口或 `pipelines/a_share/runner.py`。

## API 和平台

FastAPI layer 聚合本地 run roots，提供 run list、evidence payload、CSV preview、direct artifact download、ZIP download、comparison、pipeline snapshot/run 和 Factor Factory routes。

API-visible run identity 是 `(source_alias, run_id)`。Run list、evidence routes、CSV preview、direct downloads、ZIP cache、comparison 和 React state 都必须保留 `source_alias`，不能只按 `run_id` 折叠。

边界：

- API 是本地研究辅助，不是部署服务。
- `POST /api/runs` 只从用户提供的本地输入构造 `EvaluationSpec`。
- React frontend 只消费 API 返回的数据模型，不直接读本地文件，不重新计算 IC、RankIC、weights 或 turnover。
- Factor/Pipeline Studio 只发起受控 pipeline 请求，不连接外部数据源或交易接口。
- Factor Factory 前端只展示 API / artifact 结果，不在前端重新计算评价指标。

## Artifact 合同

Evaluation set 输出：

```text
manifest.json
config.json
lineage.json
summary.json
dimensions/combinations.csv
report.md
metrics/
portfolio/
diagnostics/
breakdowns/year/
breakdowns/quarter/
breakdowns/month/
```

`manifest.json` 和 `summary.json` 均包含 `artifact_schema_version: "0.6"` 与 `artifact_type: "evaluation_set"`。Manifest 中的 artifact 路径必须是 portable relative paths，不写入 Windows drive letter、用户目录或绝对路径。Public-downloadable `config.json` 和 `lineage.json` 必须在 writer、direct download 和 selected ZIP 路径上统一脱敏，不暴露 private data path 或本机绝对路径。

`manifest.display` 是 API、下载和 React workbench 的展示合同。新增 artifact 时，writer / loader、API payload/download、frontend display consumers、tests 和 docs 必须同步。

## Phase 2 边界

Phase 1 中 `interfaces/api` 是本地 FastAPI 适配层。未来 `apps/api` 是可部署 HTTP 应用装配层，两者职责不同，不在当前阶段创建或合并。

平台边界：

- API 是本地研究辅助，不是部署服务。
- `POST /api/runs` 只从用户提供的本地输入构造 `EvaluationSpec`。
- 删除接口只能按项目规则处理明确允许删除的本地 uploaded artifacts。
- React frontend 只消费 API 返回的数据模型，不直接读本地文件，不重新计算 IC、RankIC、weights 或 turnover。
- Factor/Pipeline Studio 只发起受控 pipeline 请求，不连接外部数据源或交易接口。
- Factor Factory 前端浏览只展示 API / artifact 结果，不在前端重新计算 IC、cumulative returns、weights 或 turnover。
- 已完成 artifact ZIP packaging 下沉到 `artifacts/`。
- 后续仍需单独处理的架构债包括 frontend feature split、eligibility / tradability 诊断可视化、risk/regime/cost/capacity 扩展和 ML prediction handoff。

## Comparison

Comparison table 只提供独立指标排名，不生成综合评分：

- `rank_mean_ic`、`rank_mean_rankic`、`rank_mean_final_return`：数值越高排名越靠前。
- `rank_mean_turnover`、`rank_total_cost`：数值越低排名越靠前。
- 不提供 `overall_score`，避免把不同研究目标混成单一主观评价口径。

上游调用方接入契约见 [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md)。
