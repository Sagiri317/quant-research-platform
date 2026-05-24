# Evaluation / Artifacts / API 地图

本文记录当前 evaluation layer、artifact schema `0.5`、`manifest.display`、FastAPI evidence API、downloads 和 React workbench 的接口事实。

## 总链路

```text
market module or local adapter
-> EvaluationSpec
-> run_evaluation
-> evaluate_to_payload
-> DataFrame/Tensor kernels
-> write_evaluation_set_artifacts
-> schema 0.5 evaluation-set run
-> load_evaluation_set / run_repository
-> FastAPI evidence/download/comparison APIs
-> React evidence workbench
```

## Evaluation Public Contract

| 对象 | 代码位置 | 职责 |
|---|---|---|
| `EvaluationSpec` | `src/quant_research_platform/evaluations/evaluation_spec.py` | 一次 evaluation set 的输入合同。 |
| `WeightConfig` | `evaluation_spec.py` | 权重方法、方向、组合模式、分位阈值和 risk column。 |
| `ExecutionConfig` | `evaluation_spec.py` | execution lag 和 execution price column metadata。 |
| `CostConfig` | `evaluation_spec.py` | 成本配置。 |
| `TradabilityConfig` | `evaluation_spec.py` | eligibility / can_enter / can_exit column 配置。 |
| `NeutralizationConfig` | `evaluation_spec.py` | group demean 或 OLS neutralization 输入。 |
| `TimeSemanticsConfig` | `evaluation_spec.py` | decision、label、as-of 时间语义。 |
| `TensorBatchConfig` | `evaluation_spec.py` | tensor engine batch layout、backend、dtype 和 device 配置。 |
| `ProfileConfig` | `runtime/profiling.py` | evaluation profiling 开关和输出配置。 |
| `EvaluationRunPayload` | `evaluation_payload.py` | artifact 写盘前的内部 payload 边界。 |
| `EvaluationSetResult` | `evaluation_result.py` | `run_evaluation` 返回对象。 |
| `run_evaluation(spec)` | `evaluation_runner.py` | 对外稳定执行入口。 |
| `evaluate_to_payload(spec)` | `evaluation_runner.py` | 纯计算入口，不写 artifacts。 |

Public import 仍保持：

```python
from quant_research_platform import EvaluationSpec, WeightConfig, run_evaluation
```

## Panel 输入要求

long-format panel 必须有：

| 角色 | 典型列 | 要求 |
|---|---|---|
| timestamp | `timestamp`, `trade_date`, `bar_time` | 可解析为 datetime。 |
| asset | `asset`, `symbol`, `stock_code` | 非空字符串，股票代码不能丢前导零。 |
| feature | `mom_20d`, `prediction` | numeric。 |
| label | `forward_return_5d`, `target_return` | numeric。 |
| PnL return | `forward_return_1d` | 用于 portfolio return path。 |

可选列：

- `segment_columns`
- `risk_column`
- `eligibility_column`
- `can_enter_column`
- `can_exit_column`
- neutralization groups / exposures

`label_columns` 与 `pnl_return_column` 不同：

- labels 用于 IC、RankIC、grouped returns、spread diagnostics。
- PnL return 用于 portfolio returns、PnL、performance summary、time breakdowns。

## Artifact Contract

Evaluation runs 必须是：

```text
artifact_type = "evaluation_set"
artifact_schema_version = "0.5"
```

核心文件：

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
```

启用 `time_breakdowns` 后会按配置出现可选目录：

```text
metrics_by_year/
metrics_by_quarter/
metrics_by_month/
```

`comparison.csv` 不是单个 evaluation run 的物理必备文件。它是 `manifest.display.downloads` 暴露的动态 index item，由 API comparison 或 `write_run_comparison` 按多个 completed runs 生成。

核心模块：

| 模块 | 关键对象/函数 | 职责 |
|---|---|---|
| `artifact_schema.py` | `ARTIFACT_SCHEMA_VERSION`, `EVALUATION_SET_ARTIFACT_TYPE`, `build_evaluation_set_artifact_paths` | schema version、required tables、path contract。 |
| `artifact_type_registry.py` | `ArtifactTypeContract`, `EVALUATION_SET_CONTRACT` | 轻量 artifact type registry。 |
| `display_manifest.py` | `build_display_manifest` | 构造 `manifest.display`。 |
| `manifest_validation.py` | `validate_artifact_schema_version`, `validate_display_manifest`, `validate_evaluation_set_table_columns` | schema/display/path/table/chart validation。 |
| `artifact_writer.py` | `ArtifactOutputs`, `write_evaluation_set_artifacts`, `build_evaluation_set_manifest` | 写 evaluation-set artifacts。 |
| `artifact_loader.py` | `LoadedEvaluationSet`, `load_evaluation_set` | 严格读取 completed run。 |
| `artifact_inventory.py` | `index_evaluation_sets` | 离线 run inventory。 |
| `artifact_comparison.py` | `compare_evaluation_sets`, `write_run_comparison` | 离线 run comparison。 |
| `artifact_package.py` | `build_selected_artifact_zip`, `require_manifest_display_download_item`, `resolve_manifest_display_artifact_path` | manifest.display download gate、direct artifact path resolution、selected ZIP file collection、dynamic item handling、cache fingerprint、ZIP writing。 |

明确不支持：

- 旧 `artifacts.xlsx`。
- 旧 `total_gross_portfolio_return` / `total_net_portfolio_return` 字段。
- unsupported schema fallback。
- 绝对路径或 Windows drive letter 写入 manifest artifact paths。

## `manifest.display`

`manifest.display` 是唯一展示合同，覆盖：

- table preview。
- chart source tables。
- direct download items。
- ZIP download items。
- download presets。

API 和 frontend 不允许绕过它访问隐式文件。

校验要求：

- `display.schema_version == "0.5"`。
- `display.tables` 必须覆盖 artifact CSV tables。
- 每个 display table 必须有 matching download item。
- download preset 只能引用已暴露 items。
- path 必须是 portable relative POSIX path。
- chart columns 必须存在于对应 table。

## API Services

| Service | 代码位置 | 职责 |
|---|---|---|
| `ApiContext` | `interfaces/api/services/context.py` | app state 中的 run roots、pipeline root、Factor Factory root 和 cache root resolver。 |
| `run_repository.py` | `interfaces/api/services/run_repository.py` | run roots、run discovery、run index、loader wrapper、uploaded-run delete。 |
| `download_service.py` | `interfaces/api/services/download_service.py` | direct download、report download 和 selected ZIP 的 HTTP response adapter。 |
| `comparison_service.py` | `interfaces/api/services/comparison_service.py` | API-visible comparison JSON/CSV。 |
| `pipeline_service.py` | `interfaces/api/services/pipeline_service.py` | factor templates、pipeline snapshots、`start_from` pipeline run dispatch。 |
| `upload_service.py` | `interfaces/api/services/upload_service.py` | CSV upload -> `EvaluationSpec` -> evaluation run。 |

`run_repository` 身份规则：

```text
RunRef = (source_alias, run_id, path)
```

所有 run list、evidence routes、CSV preview、direct downloads、ZIP cache、comparison 和 React state 都必须保留 `source_alias`。不能只用 `run_id`。

## API Routes

| Route file | Endpoint group | 说明 |
|---|---|---|
| `routes/health.py` | `/health` | 健康检查和安全 root 摘要。 |
| `routes/markets.py` | `/markets`, `/api/markets` | 市场能力；A 股 enabled，其他 planned/disabled。 |
| `routes/pipeline.py` | `/api/factor-templates`, `/api/pipeline-snapshots`, `/api/pipeline-runs` | templates、snapshot 状态、`start_from` runner 状态。 |
| `routes/runs.py` | `/api/runs`, evidence, table previews | run list、删除 uploaded run、evidence、metrics/portfolio/diagnostics/breakdowns preview。 |
| `routes/artifacts.py` | `/api/runs/{source_alias}/{run_id}/artifacts/{artifact_item:path}` | display-exposed artifact direct download。 |
| `routes/downloads.py` | `/report.md`, `/download` | report 和 selected ZIP download。 |
| `routes/comparison.py` | `/api/comparison`, `/api/comparison.csv` | API-visible runs comparison。 |
| `routes/uploads.py` | `/api/templates/factor-csv`, `POST /api/runs` | CSV upload 保留入口。 |
| `routes/factor_factory.py` | `/api/factor-factory-runs` | Factor Factory artifact `0.1` 浏览和下载。 |

## Download Contract

Direct download 和 ZIP download 必须满足：

- requested item in `manifest.display.downloads.items`。
- path resolves inside run directory。
- item path is safe relative path。
- ZIP cache identity includes `source_alias` and `run_id`。
- `comparison.csv` is generated dynamically from API-visible runs when requested。
- unknown item is rejected before writing cache。
- removed workbook `artifacts.xlsx` returns 404 / not exposed。

ZIP packaging rules live in `artifacts/artifact_package.py`; API routes only parse requests, resolve API-visible dynamic items such as `comparison.csv`, and return HTTP responses.

## Comparison Contract

Comparison 提供独立指标排名，不提供综合评分：

- `rank_mean_ic`
- `rank_mean_rankic`
- `rank_mean_final_return`
- `rank_mean_turnover`
- `rank_total_cost`

`qrp compare` uses `write_run_comparison` offline. `/api/comparison` uses API-visible run indexes. These are related but not the same call path.

## React Workbench

| 文件 | 职责 |
|---|---|
| `web/src/shell/App.tsx` | Workbench shell、view state、top-level layout。 |
| `web/src/shared/api/api.ts` | API types and request helpers。 |
| `web/src/features/runs/useRuns.ts` | run list fetch。 |
| `web/src/features/evidence/useRunEvidence.ts` | evidence fetch。 |
| `web/src/features/evidence/runEvidenceState.ts` | run/combination switching state helpers。 |
| `web/src/features/evidence/ResultsView.tsx` | summary、tables、charts、report view composition。 |
| `web/src/features/evidence/CsvBrowser.tsx` / `web/src/features/uploads/CsvPreviewPanel.tsx` | API table preview。 |
| `web/src/features/evidence/RunCharts.tsx` | display-driven charts。 |
| `web/src/features/evidence/ArtifactDownloads.tsx` | display-driven direct/ZIP downloads。 |
| `web/src/features/evidence/RunComparison.tsx` | comparison UI。 |
| `web/src/features/uploads/CreateRunPanel.tsx` | CSV upload local entry。 |
| `web/src/features/factor-studio/FactorStudio.tsx` | factor expression draft。 |
| `web/src/features/pipeline/PipelineBuilder.tsx` | pipeline snapshot request UI。 |
| `web/src/features/factor-factory/FactorFactoryRuns.tsx` | Factor Factory run browser。 |
| `web/src/shared/factorDraft.ts` | Factor Studio 和 Pipeline Builder 共享的 factor draft state type/default。 |

前端结构：

- `web/src/shell/` 只负责 workbench 组合、view mode 和跨 feature 协调。
- `web/src/features/*/` 负责 feature-local UI、hooks、state helpers 和 tests。
- `web/src/shared/*` 负责 API types/helpers、共享 feature state types、通用 UI、formatting、downloads 和 test fixtures。
- 依赖方向是 `shell -> features -> shared`；`shared/*` 不反向依赖 feature 或 shell。
- 跨 feature import 应保持少量；优先通过 shell 组合，或把真正共用的 helper 上移到 `shared/`。
- 不急于增加 broad barrel exports；只有 feature public surface 稳定后才做有意筛选的 export。

Frontend rules:

- 消费 API 和 `manifest.display`。
- 不直接读本地文件。
- 不重新计算 IC、RankIC、positions、turnover、PnL 或 cost。
- run switching / combination switching 不能显示 stale evidence。
- download picker 和 ZIP preset 必须来自 display contract。

## Input Adapters

| Adapter | 入口 | 用途 |
|---|---|---|
| A 股 adapter | `run_factor_evaluation.py`, `evaluation_adapter.py` | A 股真实私有 processed factor run。 |
| DataFrame/Python | 手动构造 `EvaluationSpec` | market module 或本地脚本直接调用。 |
| Parquet helper | `make_parquet_evaluation_spec()` | 上游已有 local Parquet panel。 |
| Parquet CLI | `qrp evaluate-parquet` | 外部/local Parquet adapter check。 |
| CSV upload | `POST /api/runs` | 本地调试/兼容入口。 |

## Verification Surface

当前合同变更后优先跑：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/evaluations tests/artifacts tests/interfaces/api
conda run -n analysis python -m ruff check src tests
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
git diff --check
```

需要覆盖的负例：

- unsupported artifact schema。
- missing display table/download。
- unknown download item。
- unsafe artifact path。
- lost `source_alias`。
- retired `total_*_portfolio_return` fields。
- removed `artifacts.xlsx` exposure。
- stale frontend evidence after run/combination switching。
