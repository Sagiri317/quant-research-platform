# Quant Research Platform

本项目当前阶段是 Local Research Platform：本地多市场量化研究平台。当前第一个完整 market module 是 A 股：它把用户已批准的本地 CSMAR 数据转换为可复现的 interim 数据、research panel、factor table、pipeline snapshots 和 evaluation artifacts，并在同一项目内提供 evaluation layer、artifact contract、FastAPI evidence API 和 React evidence workbench。

本项目类型：`research platform / local quant research infrastructure`。

当前边界：本项目不是投资建议、实盘交易、paper trading、生产交易系统或部署服务；不登录 CSMAR，不自动下载数据，不连接券商、交易所或外部行情 API，不处理账号、密码、cookie、token 或 `.env`。

数据阶段：A 股使用本地已批准离线真实数据；CSMAR raw / interim / processed 和 `reports/private_real/` 均为私有边界，不提交、不公开。

## 研究问题

- 如何把本地私有 A 股数据整理为可复现、可审计、可扩展的 research platform artifact？
- 如何用统一的 evaluation contract、`manifest.display`、API/download/frontend 合同支撑 factor evaluation、Pipeline Builder 和 Factor Factory？
- 如何在不暴露私有数据、不连接外部 API、不暗示交易能力的前提下，保留可验证的研究证据链？

## 方法概览

```text
local approved data
-> market module
-> research panel / factor table
-> evaluation layer
-> schema 0.7 artifacts
-> manifest.display
-> API / downloads / evidence workbench
-> report and research evidence
```

当前能力：

- A 股是 enabled market module，数据源边界是用户手工导出的本地 CSMAR 文件。
- Evaluation layer、artifact writer/loader、schema `0.7`、diagnostics framework、`manifest.display`、API、下载和 workbench 已内置到 `quant_research_platform`。
- Factor Studio / Pipeline Builder / Data Checkpoints 是平台主流程；CSV upload 只作为 legacy/dev uploaded-panel smoke / compatibility 入口保留。
- Pipeline Builder 前端默认生成配置快照，调用 `POST /api/pipeline-snapshots`；也可切换到执行模式，调用 `POST /api/pipeline-runs`。
- 可执行 pipeline 调用的公开控制参数是 `kernel` 和 `start_from`，支持 `existing_factors`、`rebuild_factors`、`rebuild_panel`、`rebuild_interim`，并且一定运行到 evaluation。
- Pipeline failed-run reliability MVS 已合入 `main`：persisted status fallback、failed checkpoints consistency、stable `failure_code`、sanitized failed payload 和 partial failed evaluation guard 已完成；retry/resume/overwrite 仍未实现。
- Factor Factory 已有后端 registry、批量评估、API 浏览入口和本地 workbench 浏览入口；candidate summary 按 `candidate_id + combination_id` 展开 N7 evaluation results。
- Evidence workbench 只展示 artifact/API 结果，不重新计算 eligibility、tradability、IC、returns、weights 或 turnover；图表通过 bounded preview pagination 拼接当前组合时间轴，CSV 明细仍保持 preview row limit。
- Crypto 和其他市场只显示为 planned / disabled，不代表真实数据、交易或空实现包。

## 核心结果

| 发现 | 证据 | 状态 |
|---|---|---|
| 平台合同已迁入 `quant_research_platform` 命名空间 | schema `0.7` artifacts、diagnostics registry、`manifest.display`、FastAPI evidence API、download contract 和 React evidence workbench | 已验证 |
| A 股 pipeline 已采用 snapshot/run split | `POST /api/pipeline-snapshots` 只保存配置；`POST /api/pipeline-runs` 用 `start_from` 执行到 evaluation | 已验证 |
| N1/N2/N3/N4/N7/N8 已合入 `main` | Pipeline runner、tradability diagnostics、ZIP packaging、frontend feature split、run identity、contract consumer convergence | 已验证 |
| 当前 2023-2025 evidence 可作为回归入口 | `schema07_eval_2023_2025_v1`、`schema07_eval_2023_2025_rev120_v1` 和 Factor Factory `rev_120d` smoke | 已验证 |
| N5 MVS descriptive proxy diagnostics 已合入 `main`，N5 private real-data smoke 已通过；pipeline failed-run reliability MVS 已合入 `main` | run id `n5_private_smoke_2025h2_rev120_v1`；2025H2 aggregate smoke；schema `0.7` validation、diagnostics tables、`manifest.display`、API preview 和 selected ZIP smoke 已通过；pipeline failed-run commit `67f363b550f6c3ce04be68fb30a47ed3088eb32d`；true risk / regime / capacity / simulation、retry/resume/overwrite 和 global API error envelope migration 仍是未来工作 | smoke passed / MVS merged |

## 复现方式

常用后端检查：

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests
```

前端检查在 `web/` 下运行：

```powershell
npm test -- --run --maxWorkers=1
npm run lint
npm run build
```

本地 API + React workbench：

```powershell
scripts\start_private_workbench.ps1
scripts\stop_workbench.ps1
```

当前阶段：Local Research Platform。Phase 1 已完成是历史验收状态；N1/N2/N3/N4/N7/N8 merged to `main` 是已完成事实；后续 N5/N6 独立推进，不能让 roadmap 成为当前实现范围或引入 speculative infrastructure。

最近一次检查见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。

## Codex Task Quick Start

Codex 开始新任务时，应先读 [AGENTS.md](AGENTS.md)，判断任务模式，并按任务表面加载最小必要上下文。普通 coding task 不默认加载未来架构目标；只有 Decision Task、architecture planning 或 future capability boundary task 才默认读取 `docs/architecture/target-architecture-and-capabilities.md`。

Medium+ task 先输出 task brief、implementation plan、scope declaration、stop conditions 和 verification plan，等用户批准后再实现。涉及 artifact schema、`manifest.display`、API/download、frontend evidence、A 股 pipeline、Factor Factory 或 generated runs 时，按 `.codex/skills/quant-research-platform-maintenance/` 的 context routing 和 contracts reference 加载当前合同。

常用 Codex 任务模板见 `.codex/templates/`；普通任务优先选 fast-path / focused 模板，`/goal`、overnight 和 review-and-fix-loop 使用 `goal-overnight.md`。

本项目的 historical plans/specs 已归档到 `docs/archive/superpowers/`；它们可用于追溯历史决策，但不是当前合同真相源。当前合同以源码、测试、`manifest.display`、根目录文档、`docs/architecture/`、`docs/code-map/` 和项目维护 skill 为准。

## 当前结构

```text
src/quant_research_platform/
  evaluations/           EvaluationSpec, run_evaluation, panel contract
  kernels/               DataFrame / tensor evaluation helpers
  artifacts/             schema 0.7 writer, loader, registry, display contract
  interfaces/api/        local FastAPI evidence API, downloads, run repository
  interfaces/cli/        qrp command entry
  datasets/              generic CSV / Parquet readers
  markets/implementations/a_share/
                         CSMAR-backed A-share market module
  pipelines/a_share/     snapshots, start_from runner, checkpoints
  research/              Factor Factory and future research contracts

web/                     React/Vite evidence workbench
config/a_share/          A-share factor templates and pipeline configs
scripts/                 local workbench launch/stop scripts and private run roots
data/                    local private data boundary
reports/private_real/    local private real-data outputs
docs/data/               data documentation and field availability
docs/code-map/           source reading map
```

研究项目配套文档：

```text
REPORT.md               正式研究报告
RESEARCH_CARD.md        给作者和 Codex 使用的证据底稿
PROJECT_STATUS.md       本地维护状态、任务、想法和决策
AGENTS.md               Codex 安全规则
```

## A 股 Pipeline 接口

Snapshot 入口只保存配置：

```http
POST /api/pipeline-snapshots
```

Run 入口执行 pipeline：

```http
POST /api/pipeline-runs
```

`start_from` 定义从哪一层开始，结束层固定为 evaluation：

```text
existing_factors -> evaluation
rebuild_factors  -> build_factors -> evaluation
rebuild_panel    -> build_panel -> build_factors -> evaluation
rebuild_interim  -> build_interim -> build_panel -> build_factors -> evaluation
```

状态和 checkpoints 分别通过对应资源读取：

```http
GET /api/pipeline-snapshots/{run_id}
GET /api/pipeline-snapshots/{run_id}/checkpoints
GET /api/pipeline-runs/{run_id}
GET /api/pipeline-runs/{run_id}/checkpoints
```

## A 股专用入口

A 股真实私有数据流程优先使用 market module 自身入口，不要为了日期过滤临时绕到通用 `evaluate-parquet`。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"

conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.datasets.build_panel `
  --start-year 2014 `
  --end-year 2025 `
  --end-date 2025-12-31

conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.factors.build_factors `
  --start-year 2014 `
  --end-year 2025 `
  --windows 20,60,120 `
  --horizons 1,5,10,20,60,120

conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation `
  --start-year 2023 `
  --end-year 2025 `
  --start-date 2024-01-01 `
  --end-date 2024-12-31 `
  --features mom_20d vol_20d avg_amount_20d rev_120d `
  --labels forward_return_5d forward_return_20d `
  --portfolio-modes long_only `
  --kernels dataframe `
  --report-root reports/private_real/factor_evaluation/example_2024_dataframe_long_only
```

## Artifact 和 Workbench 合同

当前新生成的 evaluation artifacts 使用 schema `0.7`。一个 API-visible evaluation run 可以包含多个 `combination_id`，组合维度由 `dimensions/combinations.csv` 管理；diagnostics policy 维度由 `dimensions/diagnostic_policies.csv` 管理。

成本、权重方法、benchmark、universe、eligibility、tradability、生成环境和输入 lineage 等 run-level 配置写入 `run_context.json`，不进入 combinations 维度表；portable artifacts 不再写出 `config.json` 或 `lineage.json`。

Schema `0.7` 的 required 输出包括 `metrics/ic.csv`、`metrics/grouped_returns.csv`、`metrics/evaluation_metrics_summary.csv`、`portfolio/returns.csv`、`input_quality/coverage_summary.csv`、`dimensions/diagnostic_policies.csv`、`diagnostics/diagnostic_summary.csv` 和 `diagnostics/diagnostic_metrics.csv`；`portfolio/weights.parquet` 与其他 diagnostics materialized views 只在请求并生成时暴露。启用 year / month breakdown 时还会分别生成 `breakdowns/year/ic.csv` 和 `breakdowns/month/ic.csv` 作为 IC / RankIC breakdown artifacts。公开合同使用 `kernel` 和 `return_source_column`；diagnostics 中的 risk/regime/cost/capacity 是 descriptive proxy diagnostics，不代表 alpha proof、真实 capacity、validated risk model、paper/live trading 或 investment advice。

Evaluation run 使用 artifact schema `0.7`。`src/quant_research_platform/artifacts/schema_registry.py` 与 tests 是字段数、grain、unique key、required_if / nullable_if、sentinel、generated files、display/download eligibility 的 source of truth。`manifest.display` 是 API、下载、CSV preview、React 表格、图表和“全选下载”的共同展示合同。本地 API 支持多个 run root；前端和下载接口中的 run 身份是 `(source_alias, run_id)`，不能只按 `run_id` 折叠或缓存。

N5 v1 当前状态是 partial implementation + contract groundwork：已通过 schema registry、required diagnostics tables 和 conditional materialized views 承载 post-run descriptive proxy diagnostics。risk/regime/cost/capacity 仍只是不用于决策的 proxy diagnostics，不是正式 risk model、capacity model、regime engine、backtest 或 simulation system。

N5 private real-data smoke 已通过：run id `n5_private_smoke_2025h2_rev120_v1`，数据范围只记录为 2025H2 aggregate smoke。Schema `0.7` validation passed；`diagnostic_summary`、`diagnostic_metrics`、`scenario_metrics`、`exposure_capacity_summary`、`portfolio_risk_summary`、`manifest.display`、API preview 和 selected ZIP smoke 已检查通过。Workbench browser acceptance 本轮未运行；API/display evidence 足以证明 generic diagnostics table visibility。

当前保留的 2023-2025 private canonical 样例：

- `reports/private_real/factor_evaluation/schema07_eval_2023_2025_v1` 覆盖 `mom_20d, vol_20d, avg_amount_20d, rev_120d x 2 labels x dataframe x long_only = 8` 个 combinations。
- `reports/private_real/factor_evaluation/schema07_eval_2023_2025_rev120_v1` 覆盖 `rev_120d x 2 labels x 2 kernels x 2 portfolio modes = 8` 个 combinations，用于 120 日反转的 kernel / portfolio mode parity smoke。
- `reports/private_real/ff/20260527_rev120_factory_smoke_v1` 是当前 Factor Factory `rev_120d` smoke 样例。

新增或修改 artifacts 时必须同时检查：

- writer / loader 是否仍只接受当前 schema；
- `manifest.display.tables`、downloads、presets、charts 是否覆盖实际展示项；
- API evidence、direct download、ZIP download 是否只暴露 display contract 中的条目；
- 前端 run switching / combination switching 是否不会显示旧状态；
- docs 和 code map 是否同步。

## 私有数据边界

以下目录不提交、不公开：

```text
data/raw_private/
data/interim_private/
data/processed_private/
reports/private_real/
```

CSMAR raw/interim/processed 数据、可还原 vendor 明细、私有真实报告和任何凭据都不得复制到公开材料、GitHub、issue、PR 或聊天记录中。公开材料只允许使用代码、方法说明和脱敏聚合结论，并且必须避免写成真实收益承诺或投资建议。

## 限制

- 当前完整 market module 只有 A 股；crypto 和其他市场只是 planned / disabled 能力表述，不保留空实现包。
- 当前只覆盖第一批 price-volume factors。
- N5 full risk/regime/cost/capacity capability 与 N6 ML prediction handoff 仍需单独设计和验证；当前 N5 v1 只代表 descriptive proxy diagnostics 和 artifact/display contract groundwork。
- 模拟盘不属于当前范围；未来如果做，必须保持与真实交易接口隔离，并单独审批。
- CSMAR 明细数据和可还原 vendor 数据的报告不公开。

## 阅读顺序

1. [README.md](README.md)：项目公开入口。
2. [REPORT.md](REPORT.md)：完整研究叙事。
3. [RESEARCH_CARD.md](RESEARCH_CARD.md)：证据底稿。
4. [PROJECT_STATUS.md](PROJECT_STATUS.md)：本地维护状态。
5. [ARCHITECTURE.md](ARCHITECTURE.md)：平台架构。
6. [docs/architecture/ai-native-engineering-principles.md](docs/architecture/ai-native-engineering-principles.md)：AI-native 长期工程原则。
7. [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md)：evaluation 输入输出合同。
8. [docs/code-map/README.md](docs/code-map/README.md)：源码阅读地图。
9. [docs/data/a-share-csmar-data-manual.md](docs/data/a-share-csmar-data-manual.md)：A 股 CSMAR 数据说明。
10. [docs/data/a-share-field-inventory.md](docs/data/a-share-field-inventory.md)：A 股字段可用性清单。
11. [AGENTS.md](AGENTS.md)：Codex 规则。
