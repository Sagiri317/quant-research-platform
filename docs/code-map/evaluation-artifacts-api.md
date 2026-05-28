# Evaluation / Artifacts / API 地图

## N7 当前合同快照（schema 0.7）

当前新生成的 evaluation-set artifact 使用 schema `0.7`：

```text
market adapter / upload / parquet helper
-> EvaluationSpec(kernel=..., return_source_column=...)
-> evaluate_to_payload / run_evaluation
-> write_evaluation_set_artifacts
-> schema 0.7 evaluation_set artifact
-> manifest.display
-> API evidence/download/preview/comparison
-> React evidence workbench
```

核心路径：

- `dimensions/combinations.csv` 是动态维度表；第一阶段 required columns 仅为
  `combination_id, feature, label, kernel, portfolio_mode`。
- fact tables 通过 `combination_id` join 维度表；run-level constant 配置写入
  `run_context.json`。
- required portfolio outputs 包括 `portfolio/weights.parquet` 和
  `API-derived primary_cumulative_return chart from portfolio/returns.csv`；不再写 schema `0.7` 的 `pnl.csv`。
- API/frontend 合同使用 `kernel` 和 `return_source_column`。
- `compare_evaluation_sets` 是低内存 summary comparison；更详细的
  `write_evaluation_comparison_report()` 只输出 CSV/JSON/Markdown，不进入主网页。
- Table preview 由 evidence API 提供 bounded preview 能力，frontend 通过
  API evidence payload 中的 `manifest.display.tables[].preview_url` 和
  portable preview descriptor 消费，不从 artifact path 重新推导 route。
- Public-downloadable metadata files, currently `run_context.json`,
  are sanitized consistently when written, directly downloaded, and packaged in
  selected ZIPs; selected ZIP cache fingerprints hash the sanitized dynamic
  metadata bytes.

本文记录当前 evaluation layer、artifact schema `0.7`、`manifest.display`、FastAPI evidence API、downloads 和 React workbench 的接口事实。

## 总链路

```text
market module or local adapter
-> EvaluationSpec
-> run_evaluation
-> evaluate_to_payload
-> DataFrame/Tensor kernels
-> write_evaluation_set_artifacts
-> schema 0.7 evaluation-set run
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
| `TensorBatchConfig` | `evaluation_spec.py` | tensor kernel batch layout、backend、dtype 和 device 配置。 |
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
| return source | `forward_return_1d` | 用于 portfolio return path。 |

可选列：

- `segment_columns`
- `risk_column`
- `eligibility_column`
- `can_enter_column`
- `can_exit_column`
- neutralization groups / exposures

`label_columns` 与 `return_source_column` 不同：

- labels 用于 IC、RankIC、grouped returns、spread diagnostics。
- return source 用于 portfolio returns、cumulative returns、performance summary、time breakdowns。

## Artifact Contract

Evaluation runs 必须是：

```text
artifact_type = "evaluation_set"
artifact_schema_version = "0.7"
```

核心文件：

```text
manifest.json
run_context.json
summary.json
run_context.json
dimensions/combinations.csv
report.md
metrics/ic.csv
metrics/evaluation_metrics_summary.csv
metrics/grouped_returns.csv
metrics/evaluation_metrics_summary.csv
metrics/evaluation_metrics_summary.csv
portfolio/weights.parquet
portfolio/returns.csv
API-derived primary_cumulative_return chart from portfolio/returns.csv
diagnostics/tradability_diagnostics.csv
input_quality/coverage_summary.csv
input_quality/coverage_summary.csv
input_quality/coverage_summary.csv
summary.json plus metrics/evaluation_metrics_summary.csv
```

启用 `time_breakdowns` 后会按配置出现可选目录：

```text
breakdowns/year/
breakdowns/year/ic.csv
breakdowns/quarter/
breakdowns/month/
breakdowns/month/ic.csv
```

`comparison.csv` 不是单个 evaluation run 的物理必备文件。它是 `manifest.display.downloads` 暴露的动态 index item，由 API comparison 或 `write_run_comparison` 按多个 completed runs 生成。

核心模块：

| 模块 | 关键对象/函数 | 职责 |
|---|---|---|
| `artifact_schema.py` | `ARTIFACT_SCHEMA_VERSION`, `EVALUATION_SET_ARTIFACT_TYPE`, `build_evaluation_set_artifact_paths` | schema version、required tables、path contract。 |
| `artifact_type_registry.py` | `ArtifactTypeContract`, `EVALUATION_SET_CONTRACT` | 轻量 artifact type registry。 |
| `display_manifest.py` | `build_display_manifest` | 构造 `manifest.display`。 |
| `display_tables.py` | `build_display_table_preview_descriptor`, `display_table_artifact_path`, `display_download_items` | display table preview/download descriptor、路径解析和 row-limit 合同 helper。 |
| `manifest_validation.py` | `validate_artifact_schema_version`, `validate_display_manifest`, `validate_evaluation_set_table_columns` | schema/display/path/table/chart validation。 |
| `public_metadata.py` | `sanitize_public_metadata` | public-downloadable metadata 脱敏；移除 backend-only input fields，并隐藏私有或本机路径。 |
| `artifact_writer.py` | `ArtifactOutputs`, `write_evaluation_set_artifacts`, `build_evaluation_set_manifest` | 写 evaluation-set artifacts。 |
| `artifact_loader.py` | `LoadedEvaluationSet`, `load_evaluation_set` | 严格读取 completed run。 |
| `artifact_inventory.py` | `index_evaluation_sets` | 离线 run inventory。 |
| `artifact_comparison.py` | `compare_evaluation_sets`, `write_run_comparison` | 离线 run comparison。 |
| `artifact_package.py` | `build_selected_artifact_zip`, `public_config_bytes`, `public_lineage_bytes`, `require_manifest_display_download_item`, `resolve_manifest_display_artifact_path` | manifest.display download gate、direct artifact path resolution、selected ZIP file collection、dynamic metadata handling、cache fingerprint、ZIP writing。 |

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
Evidence workbench 的图表可以用 preview API 读取当前组合的完整时间序列，
但必须按 `preview.max_limit` 分页，通过 `offset` / `has_more` 继续拼接；
不得为了长样本把单次 `limit` 硬编码到 display contract 之外。CSV 明细预览
仍保持 display preview row limit，完整行数据走 direct download / ZIP download。
N2 tradability diagnostics 作为 `diagnostics/tradability_diagnostics.csv`
通过 `manifest.display` 暴露；前端应通过常规 diagnostics preview/download API
消费该表，不重新计算 eligibility、can_enter 或 can_exit 计数。
Evidence workbench 的诊断卡片展示当前组合最后一行时，应使用低内存 preview
分页语义读取单行；只有首个 preview payload 返回 `total_rows_exact=true` 时，
前端才可请求 `limit=1&offset=total_rows-1&combination_id=...`。如果 row
count 非精确，卡片不得把 preview 页里的末行声明为最新诊断；CSV 明细预览仍保持
display preview row limit，不为了卡片拉完整 diagnostics 表。
新生成的 schema `0.7` evaluation-set 必须包含该表；缺少该表的旧 run 不再兼容，
需要重新生成后才能由当前 loader/API 加载。

校验要求：

- `display.schema_version == "0.7"`。
- `display.tables` 必须覆盖 artifact CSV tables。
- 每个 display table 必须有 matching download item。
- 每个 display table 必须有 portable `preview` descriptor；`preview.table_key` 必须等于 table key，`default_limit` 必须等于合同 preview row limit，`supports_combination_filter` 与 `filter_parameter` 必须一致。
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
| `factor_template_service.py` | `interfaces/api/services/factor_template_service.py` | controlled factor template catalog 和 template source metadata。 |
| `pipeline_service.py` | `interfaces/api/services/pipeline_service.py` | pipeline snapshots、`start_from` pipeline run dispatch。 |
| `table_preview_service.py` | `interfaces/api/services/table_preview_service.py` | `manifest.display.tables` gated CSV/Parquet preview、limit/offset/combination filter。 |
| `upload_service.py` | `interfaces/api/services/upload_service.py` | legacy/dev uploaded-panel CSV smoke -> `EvaluationSpec` -> evaluation run。 |

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
| `routes/factor_templates.py` | `/api/factor-templates` | factor template catalog。 |
| `routes/pipeline.py` | `/api/pipeline-snapshots`, `/api/pipeline-runs` | snapshot 状态、`start_from` runner 状态。 |
| `routes/runs.py` | `/api/runs`, evidence, table previews | run list、删除 uploaded run、evidence、generic `/tables/{table_key}/preview` 和兼容 generic table preview。 |
| `routes/artifacts.py` | `/api/runs/{source_alias}/{run_id}/artifacts/{artifact_item:path}` | display-exposed artifact direct download。 |
| `routes/downloads.py` | `/report.md`, `/download` | report 和 selected ZIP download。 |
| `routes/comparison.py` | `/api/comparison`, `/api/comparison.csv` | API-visible runs comparison。 |
| `routes/uploads.py` | `/api/templates/factor-csv`, `POST /api/runs` | CSV upload 保留为 legacy/dev uploaded-panel smoke / compatibility 入口。 |
| `routes/factor_factory.py` | `/api/factor-factory-runs` | Factor Factory artifact `0.1` 浏览和下载。 |

## Download Contract

Direct download 和 ZIP download 必须满足：

- requested item in `manifest.display.downloads.items`。
- path resolves inside run directory。
- item path is safe relative path。
- ZIP cache identity includes `source_alias` and `run_id`。
- `run_context.json` direct downloads and selected ZIP entries use
  sanitized public metadata bytes, not raw private/local path payloads。
- selected ZIP cache identity for dynamic metadata includes the sanitized content
  digest, so metadata changes do not reuse stale ZIP files。
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
| `web/src/features/evidence/CsvBrowser.tsx` / `web/src/features/uploads/CsvPreviewPanel.tsx` | API table preview；artifact inspection / audit support，不是通用 CSV viewer。 |
| `web/src/features/evidence/RunCharts.tsx` | display-driven charts。 |
| `web/src/features/evidence/ArtifactDownloads.tsx` | display-driven direct/ZIP downloads。 |
| `web/src/features/evidence/RunComparison.tsx` | comparison UI。 |
| `web/src/features/uploads/CreateRunPanel.tsx` | CSV upload local entry。 |
| `web/src/features/factor-studio/FactorStudio.tsx` | factor expression draft。 |
| `web/src/features/pipeline/PipelineBuilder.tsx` | pipeline snapshot/run request UI；run 完成后刷新 run list 并选择 `pipeline/<run_ref or run_id>`。 |
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
- Table preview 使用 API evidence payload 下发的 `preview_url` 和 `preview` metadata；不根据 display table `path` 推导 preview endpoint。
- 不直接读本地文件。
- 不重新计算 IC、RankIC、weights、turnover、cumulative returns 或 cost。
- run switching / combination switching 不能显示 stale evidence。
- download picker 和 ZIP preset 必须来自 display contract。

## Input Adapters

| Adapter | 入口 | 用途 |
|---|---|---|
| A 股 adapter/service | `run_factor_evaluation.py`, `service.py`, `evaluation_adapter.py` | A 股真实私有 processed factor run；CLI 和 pipeline runner 通过 service request/result 入口调用 adapter。 |
| DataFrame/Python | 手动构造 `EvaluationSpec` | market module 或本地脚本直接调用。 |
| Parquet helper | `make_parquet_evaluation_spec()` | 上游已有 local Parquet panel。 |
| Parquet CLI | `qrp evaluate-parquet` | 外部/local Parquet adapter check。 |
| A 股 evaluation service | `markets/implementations/a_share/evaluation/service.py` | CLI、pipeline runner 和后续本地调用共享的 A-share evaluation request/result 入口。 |
| CSV upload | `POST /api/runs` | legacy/dev uploaded-panel smoke / compatibility 入口。 |

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
