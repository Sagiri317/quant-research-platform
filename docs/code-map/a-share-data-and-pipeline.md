# A 股数据与 Pipeline Code Map

本文记录 A 股 market module、数据层和 pipeline runner 的当前代码事实。

## 数据层

```text
data/raw_private/csmar/
-> data/interim_private/csmar/
-> data/processed_private/csmar/panel_full/
-> data/processed_private/csmar/price_volume_factors_full/
-> reports/private_real/
```

边界：

- raw/interim/processed 和 private reports 都是本地私有数据，不提交、不公开。
- 代码不登录 CSMAR、不下载外部数据、不处理凭据。
- interim 是可审计规范化层；panel/factor table 是研究可用层；evaluation artifacts 是可展示证据层。

## Build 入口

| 层 | 入口 | 说明 |
|---|---|---|
| raw -> interim | `quant_research_platform.markets.implementations.a_share.csmar.build_interim` | 标准化 CSMAR raw CSV |
| interim -> panel | `quant_research_platform.markets.implementations.a_share.datasets.build_panel` | 生成 full research panel |
| panel -> factors | `quant_research_platform.markets.implementations.a_share.factors.build_factors` | 生成 slim factor and label table |
| panel -> literature-prior factor rebuild | `quant_research_platform.markets.implementations.a_share.factor_rebuild` | Feature-branch service for schema inventory, memory preflight, dry-run, explicit overwrite, optional backup, and private-output policy checks |
| features -> prediction handoff | `quant_research_platform.research.prediction_handoff` | Feature-branch backend service that trains local non-DL prediction signals, writes prediction parquet, evaluates through schema `0.7`, and emits aggregate comparison / EDA outputs |
| factors -> evaluation service | `quant_research_platform.markets.implementations.a_share.evaluation.service` | A 股 evaluation request/result service；CLI、pipeline runner 和 Factor Factory 的共享入口 |
| evaluation service -> CLI | `quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation` | A 股专用 evaluation CLI 薄包装 |
| orchestrated run | `quant_research_platform.pipelines.a_share.runner.run_a_share_pipeline` | 按 `start_from` 执行到 evaluation |

## Pipeline Contract

配置快照：

```http
POST /api/pipeline-snapshots
```

执行：

```http
POST /api/pipeline-runs
```

Feature-branch A-share research batch API additions:

```http
POST /api/a-share/factor-rebuild-runs
GET /api/a-share/factor-rebuild-runs/{run_id}
GET /api/a-share/factor-rebuild-runs/{run_id}/checkpoints
GET /api/a-share/factor-rebuild-runs/{run_id}/eda
POST /api/a-share/prediction-handoffs
GET /api/a-share/prediction-handoffs/{run_id}
GET /api/a-share/prediction-handoffs/{run_id}/checkpoints
GET /api/a-share/prediction-handoffs/{run_id}/schema
GET /api/a-share/prediction-handoffs/{run_id}/comparison
GET /api/a-share/prediction-handoffs/{run_id}/eda
```

Prediction handoff evaluations are ordinary schema `0.7` `evaluation_set`
artifacts under each handoff run's `evaluation/` child. They are discoverable as
`a_share_prediction/{run_id}` only when the API is explicitly configured with a
prediction handoff root; this keeps the default health/run-root contract stable.

`start_from` 控制开始层：

```text
existing_factors -> evaluation
rebuild_factors  -> build_factors -> evaluation
rebuild_panel    -> build_panel -> build_factors -> evaluation
rebuild_interim  -> build_interim -> build_panel -> build_factors -> evaluation
```

没有公开 build-only run，也没有结束层参数。需要从前面层开始时，后续依赖层必须执行。

## PipelineConfig

关键字段：

```text
market
run_id
date_start / date_end
factor_expression
factor_template_id
processed_factor_column
label_columns
return_source_column
portfolio_mode
cost_bps
kernel
start_from
raw_root
interim_root
processed_root
min_listing_trading_days
factor_windows
label_horizons
output_root
```

字段在 full panel、slim factor table、evaluation input 和 N5 diagnostics 中的当前可用性见 [../data/a-share-field-inventory.md](../data/a-share-field-inventory.md)。CSMAR vendor-specific 数据源、字段映射和列裁剪见 [../data/a-share-csmar-data-manual.md](../data/a-share-csmar-data-manual.md)。

Diagnostics-enabled A-share evaluation runs may preserve only the N5 MVS optional context whitelist from the slim factor table: `amount`, `turnover`, `total_market_cap`, `float_market_cap`, `avg_amount_*`, `turnover_*`, `amihud_*`, `amplitude_*`, and `market_type`. This is aggregate descriptive context for existing diagnostics tables; it does not pass through industry/sector fields, raw OHLC, listing lifecycle fields, private audit fields, or row-level private diagnostics.

`factor_expression` 可以是受控表达式；执行 run 时也允许指向已经存在的 processed factor column。`processed_factor_column` 可用于明确指定候选因子列。

## Step Registry

当前 step keys：

| Step | Inputs | Outputs |
|---|---|---|
| `snapshot` | `PipelineConfig` | `run_config.yaml`, `warnings.json`, `summary.md` |
| `build_interim` | raw CSMAR | interim data |
| `build_panel` | interim data | research panel |
| `build_factors` | research panel | factor table |
| `evaluate` | processed factor table / factor table + `PipelineConfig` | 通过 A-share evaluation service 写出 evaluation manifest、label availability 和 evaluation runs |

Runner 会写 `checkpoints.json` 和 `runtime_summary.json`。失败 checkpoint 只写稳定 failure code，不写 private path。Pipeline API 在 in-memory state 丢失后可从磁盘 `runtime_summary.json` / `checkpoints.json` 重放 sanitized failed-run status；disk replay 使用 shared private-marker sanitizer，覆盖 ordinary private layer markers 以及 `vendor_private` / `raw_vendor` 等 research-writer markers。

## Factor Factory

Factor Factory 从 `config/a_share/factor_templates/` 展开 candidates。Factor Studio 通过
`GET /api/factor-templates` 读取同一受控模板目录，template API 已独立于 pipeline
snapshot/run routes。批量评估入口：

```text
quant_research_platform.research.factor_factory.batch_runner.evaluate_existing_factor_candidates
```

它为每个 candidate 构造 `PipelineConfig(start_from="existing_factors")`，通过统一 A 股 runner 和 A-share evaluation service 评估已经存在的 factor table 列，并写出 Factor Factory artifact `0.1`。成功候选会读取 schema `0.7` 的 `dimensions/combinations.csv` 和 `metrics/evaluation_metrics_summary.csv`，在 `candidate_summary.csv` 中按 `candidate_id + combination_id` 展开表现；失败候选继续写入 sanitized failure record。

## 验证入口

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share tests/markets/implementations/a_share
conda run -n analysis python -m pytest -q tests/interfaces/api tests/research/factor_factory
```
