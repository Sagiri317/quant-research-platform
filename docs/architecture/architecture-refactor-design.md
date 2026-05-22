# Architecture Refactor Design / 架构重构设计

本文定义 `quant-research-platform` 下一轮架构重构的设计基线。目标是把当前实现推进到 [target-directory-structure.md](target-directory-structure.md) 的 Phase 1，同时保护当前已经跑通的 A 股主流程、artifact schema `0.5`、本地 API、下载和前端 evidence workbench。

本文是设计文档，不是已实现状态。逐项执行、验收和测试矩阵见 [phase-1-implementation-checklist.md](phase-1-implementation-checklist.md)。

## 1. 背景

当前迁移已经把 A 股数据流程、评价内核、schema `0.5` artifacts、FastAPI API 和 React workbench 放进 `quant_research_platform` 命名空间。A 股主链路已经清楚：

```text
CSMAR raw/interim
-> research panel
-> factor and label Parquet
-> A 股 evaluation adapter
-> EvaluationSpec
-> run_evaluation
-> schema 0.5 artifacts
-> API / workbench / downloads
```

后续能力池会覆盖 Factor Factory、真实 Pipeline Runner、风险、组合、ML、regime、多市场、专业回测、网站化和实盘等长期方向。本文只处理其中的 Phase 1 研究平台重构：先把 evaluations、artifacts、API、本地 workbench 和 A 股 pipeline runner 边界拆清楚，避免继续在现有结构上堆功能。

## 2. 重构目标

目标：

- 保持当前 public contract 稳定。
- 把 evaluations 计算和 artifact persistence 解耦。
- 把 artifacts 的 schema、display、validation、writer、loader 拆清楚。
- 把 FastAPI routes 从当前 `api/app.py` 拆到清晰的 API routes / services 边界，并按领域分组。
- 给 A 股补出真实 pipeline runner 的扩展位置。
- 为 Factor Factory、ML、risk、portfolio、regime 留出独立 research 层。
- 让前端按 Evidence、Pipeline、Factor Factory 等工作区分区，而不是让单一页面继续膨胀。

非目标：

- 不重写 A 股 CSMAR raw/interim/panel/factor 算法。
- 不升级 artifact schema 版本，除非实施中发现必须新增 schema 字段。
- 不引入数据库、队列、认证、部署、外部行情 API、券商 API、模拟盘或实盘。
- 不创建 repo-level `apps/`、`storage/`、`identity/`、`tenancy/`、`jobs/`、`workers/` 等 Phase 2 平台域。
- 不把 crypto 高频硬塞进当前日频 `EvaluationSpec`。
- 不恢复旧 evaluation 兼容逻辑或旧 generated run schema fallback。

## 3. 必须冻结的当前合同

除非用户后续明确批准破坏性变更，以下合同在重构过程中保持兼容。

### Python public API

必须保持：

```python
from quant_research_platform import EvaluationSpec, WeightConfig, run_evaluation
```

必须保持：

- `EvaluationSpec` 作为日频 / long panel factor evaluation 的主输入合同。
- `run_evaluation(spec)` 作为外部稳定入口。
- `EvaluationSetResult` 作为返回对象。
- `make_parquet_evaluation_spec()` 和 `evaluate-parquet` 作为通用本地 Parquet adapter。

允许内部变化：

- `run_evaluation(spec)` 内部可以委托给新的 `evaluate_to_payload(spec)`。
- 新增 `EvaluationRunPayload` 或 `EvaluationSetPayload`。
- 新增内部 module，例如 `evaluations/evaluation_runner.py`、`evaluations/evaluation_payload.py`、`evaluations/evaluation_result.py`、`evaluations/panel_validation.py`。

### Artifact contract

必须保持：

- `artifact_schema_version == "0.5"`。
- `artifact_type == "evaluation_set"`。
- `manifest.display` 是 API、下载、CSV preview、图表和前端表格的唯一展示合同。
- 每个 display table 都有对应 download item。
- loader 拒绝 unsupported schema，不做宽松兼容。
- 不恢复 `artifacts.xlsx`。
- 不恢复旧 `total_*_portfolio_return` summary 字段。

允许内部变化：

- `artifacts/writer.py` 可以拆成 `schema.py`、`validate.py`、`writer.py`。
- display validation 可以从 writer 中移出。
- writer 可以只接收结构化 payload，不直接理解 evaluations 执行细节。

### HTTP API contract

必须保持当前本地 API 语义：

- run identity 是 `(source_alias, run_id)`。
- run list、evidence、CSV preview、direct download、ZIP download、comparison 都保留 `source_alias`。
- direct download 和 ZIP 只暴露 `manifest.display.downloads.items` 允许的文件。
- CSV preview 只暴露 `manifest.display.tables` 允许的表。
- uploaded-run deletion 仍只能删除 `summary.data_stage == "uploaded_panel"` 且在允许 run root 内的用户上传产物。

允许内部变化：

- API app 入口只保留 `create_app()` 和 router 装配。
- 现有 endpoint handler 可以先移动到 `src/quant_research_platform/api/routes/`。
- repository、download、pipeline、comparison 逻辑可以先移动到 `src/quant_research_platform/api/services/`。

### Frontend contract

必须保持：

- 前端只消费 API 和 `manifest.display`。
- 前端不重新计算 IC、RankIC、positions、turnover、PnL 或 cost。
- run switching 和 combination switching 不显示 stale evidence。
- 下载 picker 和 ZIP preset 由 `manifest.display.downloads` 驱动。
- CSV upload 保持本地调试或兼容入口，不是主流程。
- A 股 enabled；crypto / other markets 在真实实现前只能显示 planned / disabled。

允许内部变化：

- `web/` 可以按 feature 拆分。
- CSS 可以逐步拆分，但不作为第一阶段重构的阻塞项。
- Factor Factory / Pipeline Runner 页面可以后续新增，不与 P0/P1 混做。

## 4. 目标模块结构

目标结构以 [target-directory-structure.md](target-directory-structure.md) 的 Phase 1 为准。本轮只重构本地研究平台；未来 `apps/web`、`apps/api`、`apps/workers` 和 `apps/schedulers` 是 Phase 2 网站化边界，不在本文实施范围内。

```text
web/
  evidence/
  pipeline/
  factor_factory/
  uploads/
  shared/

src/quant_research_platform/
  evaluations/
    evaluation_spec.py
    evaluation_runner.py
    evaluation_payload.py
    evaluation_result.py
    evaluation_metrics.py
    portfolio_evaluation.py
    panel_validation.py
    neutralization.py
    period_breakdowns.py
    evaluation_report.py

  kernels/
    kernel_protocol.py
    kernel_registry.py
    dataframe/
    tensor/
    gpu/
    distributed/

  artifacts/
    artifact_schema.py
    display_manifest.py
    manifest_validation.py
    artifact_writer.py
    artifact_loader.py
    artifact_inventory.py
    artifact_comparison.py
    artifact_package.py

  markets/
    common/
    asset_classes/
    instruments/
    venues/
    data_sources/
    implementations/

  research/
    factors/
    factor_factory/
    models/
    reinforcement_learning/
    portfolios/
    risk/
    regimes/
    experiments/
    validation/

  datasets/
  pipelines/
  runtime/
  interfaces/
    api/
    cli/
    sdk/
```

这个结构的关键点：

- `markets/*` 负责数据许可、字段、market rules、factor/label/risk/regime input。
- `research/*` 负责生产候选信号、模型预测、组合方案和风险诊断。
- `evaluations/*` 负责 market-neutral evaluation 和评价结果合同。
- `kernels/*` 负责 DataFrame、tensor、GPU、distributed 等底层数值计算实现。
- `artifacts/*` 负责 schema/display/persistence。
- API 边界负责把 artifacts 和 pipeline 状态暴露给本地前端；现有实现可以先在 `src/quant_research_platform/api/` 内拆 routes/services，是否迁到 `interfaces/api/` 由实施计划单独决定。
- `web/*` 负责本地用户交互，不承担研究计算；最终网站化时再迁到 `apps/web`。

## 5. Evaluations 重构设计

### 当前问题

当前 `core/evaluation.py` 同时承担：

- spec 校验；
- panel 校验；
- DataFrame / tensor engine 分派；
- table 汇总；
- summary / config / manifest 生成；
- artifact 写入；
- report 渲染；
- `EvaluationSetResult` 构造。

这让后续 Factor Factory、ML prediction、optimizer、risk diagnostics 很难复用“只计算不落盘”的能力。目标目录中这部分迁入 `evaluations/`。

### 目标设计

新增内部纯计算入口：

```python
payload = evaluate_to_payload(spec)
```

payload 结构建议：

```python
EvaluationRunPayload
  summary: dict[str, object]
  config: dict[str, object]
  manifest_seed: dict[str, object]
  tables: dict[str, DataFrame]
  report: str
```

对外入口保持：

```python
def run_evaluation(spec: EvaluationSpec) -> EvaluationSetResult:
    payload = evaluate_to_payload(spec)
    artifacts = write_evaluation_artifacts(spec.output_dir, payload)
    return EvaluationSetResult.from_payload(payload, artifacts)
```

第一阶段不要求外部用户直接使用 `evaluate_to_payload()`。它先作为内部边界，等稳定后再决定是否公开。

## 6. Artifact 重构设计

### 当前问题

当前 `artifacts/writer.py` 同时做：

- schema 常量；
- 写 JSON / CSV / Markdown；
- manifest.display 校验；
- path safety；
- chart/table/download validation。

### 目标设计

拆成：

```text
artifacts/artifact_schema.py
  ARTIFACT_SCHEMA_VERSION
  artifact table path contract
  known metadata artifacts

artifacts/display_manifest.py
  build_display_manifest

artifacts/manifest_validation.py
  validate_artifact_manifest
  validate_display_manifest
  validate_display_chart_columns
  validate_display_paths

artifacts/artifact_writer.py
  write_evaluation_set_artifacts
  write_json / write_csv / write_report

artifacts/artifact_loader.py
  load_evaluation_set
```

拆分后原则：

- writer 不再承载所有规则。
- loader 和 API 复用同一套 validation。
- 新增 artifact 时必须同步 schema、display、writer、loader、API、web 和 tests。

## 7. API 重构设计

### 当前问题

当前 `api/app.py` 包含 route 注册、run root 配置、pipeline snapshot、run list、evidence、preview、download、comparison 等多类职责。

### 目标设计

API app 入口只负责：

- 创建 FastAPI app。
- 读取 run root 和 pipeline root 配置。
- 注册 routers。
- 注入 app state 或 services。

routes 拆分：

```text
src/quant_research_platform/api/routes/health.py       health and root config
src/quant_research_platform/api/routes/markets.py      markets and capabilities
src/quant_research_platform/api/routes/pipeline.py     factor templates, pipeline run snapshots, future runner
src/quant_research_platform/api/routes/runs.py         run list, evidence, uploaded run create/delete
src/quant_research_platform/api/routes/artifacts.py    table preview and direct artifact routes
src/quant_research_platform/api/routes/downloads.py    direct report and ZIP package
src/quant_research_platform/api/routes/comparison.py   comparison JSON and CSV
src/quant_research_platform/api/routes/uploads.py      optional CSV upload helpers
```

services 拆分：

```text
src/quant_research_platform/api/services/run_repository.py
src/quant_research_platform/api/services/download_service.py
src/quant_research_platform/api/services/pipeline_service.py
```

重构时保持所有 route path 不变。先搬代码，后改设计。

## 8. A 股 Pipeline Runner 设计

当前 Pipeline Builder 只写 snapshot，不执行完整 A 股 build/evaluate。

目标 runner：

```text
PipelineConfig
-> validate
-> build_interim optional
-> build_panel
-> build_factors
-> run_factor_evaluation
-> index artifacts
-> checkpoint summary
```

第一阶段 runner 可只支持“processed 已存在”的轻量模式：

```text
factor expression/config
-> locate existing processed factor table
-> run A 股 evaluation adapter
-> write pipeline checkpoint
-> expose run in repository
```

后续再接 build panel / build factors。这样避免第一轮直接重跑大数据。

runner 必须具备：

- 不默认覆盖正式输出。
- run id path-safe。
- checkpoint 写入。
- blocking issue 和 warning 分开。
- 小数据 smoke 优先。
- 不启动 API 或前端服务。

## 9. Research 层设计

`research/` 是后续功能的扩展层，不应把 Factor Factory、ML、optimizer 都塞进 evaluations。

第一批目录：

```text
research/factor_factory/
  candidates.py
  registry.py
  evaluator.py
  reports.py

research/models/
  splits.py
  manifests.py
  predictions.py

research/portfolios/
  constraints.py
  optimizers.py
  diagnostics.py

research/risk/
  exposures.py
  covariance.py
  attribution.py

research/regimes/
  definitions.py
  diagnostics.py
```

原则：

- research 层可以生成新列、新表、新配置。
- research 层不直接重写 evaluations 的 PnL 或 IC 逻辑。
- 输出如果要进入前端，必须写成 artifact/display 可消费的结构。

## 10. Frontend 重构设计

前端后续按工作区拆分：

```text
web/src/features/evidence/
web/src/features/pipeline/
web/src/features/factor_factory/
web/src/features/uploads/
web/src/shared/
```

第一阶段不要求大规模视觉重做。优先拆：

- API types 和 request helpers。
- evidence state。
- download picker。
- run result browser。
- pipeline form。

CSS 可先保持一个文件，等页面边界稳定后再拆 feature CSS，避免同时改结构和视觉。

## 11. 分阶段实施计划

本节的 `P0-P5` 是当前重构实施阶段，不等同于 [future-capabilities.md](future-capabilities.md) 的能力池优先级 `C0-C8`。
每个阶段的具体任务、验收项、负例和测试命令见 [phase-1-implementation-checklist.md](phase-1-implementation-checklist.md)。

### P0: 合同冻结和文档基线

交付：

- 本设计文档。
- `docs/architecture/future-capabilities.md`。
- `docs/code-map/platform-flow.md` 和可拖拽 HTML。
- 删除历史迁移计划、历史迁移设计、错位 roadmap。
- 更新 status 和 code-map 入口。

验收：

- 旧文档路径无残留引用。
- `git diff --check` 通过。
- Markdown 本地链接检查通过。

### P1: Evaluation payload boundary

交付：

- 新增 `evaluations/evaluation_payload.py`。
- 新增 `evaluations/evaluation_runner.py`。
- 新增 `evaluate_to_payload(spec)`。
- `run_evaluation(spec)` 保持兼容。
- 现有 tests 通过或只做低风险断言调整。

验收：

- evaluation tests 通过。
- artifact contract tests 通过。
- A 股 adapter tests 通过。
- `run_evaluation` 仍写出 schema `0.5` artifacts。

### P2: Artifact module split

交付：

- `artifacts/artifact_schema.py`。
- `artifacts/manifest_validation.py`。
- 瘦身 `artifacts/artifact_writer.py`。
- loader、API 和 tests 共用 validation。

验收：

- artifact tests 通过。
- API display/download tests 通过。
- 搜索确认没有旧 schema fallback。

### P3: API router split

交付：

- `src/quant_research_platform/api/routes/*`。
- `src/quant_research_platform/api/services/*`。
- `src/quant_research_platform/api/app.py` 只装配 app 和 routers。

验收：

- API tests 通过。
- endpoint path 不变。
- source_alias + run_id 语义不变。
- download 和 preview 仍由 `manifest.display` 驱动。

### P4: A 股 Pipeline Runner

交付：

- `markets/implementations/a_shares/pipeline_runner.py` 或等价 Phase 1 A 股 pipeline runner。
- checkpoint model。
- pipeline API 接 runner 的轻量执行模式。

验收：

- 一年小数据、2-3 因子、2-3 label。
- dataframe / tensor 至少一条主路径先 smoke。
- long-only / long-short 覆盖。
- 不默认覆盖已有正式输出。

### P5: Factor Factory 第一版

交付：

- 候选表达式 registry。
- 小参数网格。
- 批量评价。
- failed candidate log。
- candidate summary。
- workbench 浏览入口。

验收：

- 候选来源、依赖字段、样本范围、失败原因可追踪。
- 不执行任意 Python。
- 输出进入 `manifest.display` 或专门设计的新 artifact 合同。

## 12. 测试矩阵

每阶段至少覆盖：

- `conda run -n analysis python -m pytest -q tests/evaluations tests/artifacts`
- `conda run -n analysis python -m pytest -q tests/api`
- A 股相关变更时跑 `tests/markets`
- 前端变更时在 `web/` 跑 `npm test -- --run --maxWorkers=1`
- 前端结构或下载变更时跑 `npm run lint` 和 `npm run build`
- 文档变更时跑 `git diff --check` 和本地链接检查

完整收口阶段再跑：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests
```

前端：

```powershell
Set-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
```

真实私有数据验证继续遵守小样本优先，不直接跑全量。

## 13. 风险和控制

主要风险：

- 拆 evaluations 时改变结果表列或组合维度。
- 拆 artifacts 时破坏 `manifest.display`。
- 拆 API 时 route path 或 `source_alias` 语义丢失。
- 前端拆分时出现 stale evidence 或下载项不一致。
- pipeline runner 误覆盖私有产物或占用过多内存。

控制方式：

- 每阶段先锁合同，再移动代码。
- 保持 public entrypoint 兼容。
- 每阶段只做一个主要边界。
- 用 tests 验证 absence of retired behavior。
- 小数据 smoke 优先。
- 私有 run 和 generated outputs 不提交。

## 14. 决策点

实施前需要用户确认：

- 是否接受 `evaluate_to_payload(spec)` 作为 evaluations 内部中间边界。
- 是否接受 API route path 不变、内部 routers 可移动。
- 是否接受 P4 runner 先从“processed 已存在”轻量模式开始，而不是第一版就全量 build raw/interim/panel/factors。
- 是否接受 Factor Factory 第一版只用受控 registry 和小参数网格，不接任意 Python。

推荐默认答案均为“接受”。这条路径最少返工，也最能保护当前已经通过的 A 股主流程和前端下载合同。
