# Quant Research Platform

本项目当前阶段是 Local Research Platform：本地多市场量化研究平台。当前第一个完整 market module 是 A 股，它把用户已批准的本地 CSMAR 数据转换为可复现的 interim 数据、research panel、factor table、pipeline snapshots 和 evaluation artifacts，并在同一项目内提供 evaluation layer、artifact contract、FastAPI evidence API 和 React evidence workbench。

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
-> schema artifacts
-> manifest.display
-> API / downloads / evidence workbench
-> report and research evidence
```

当前能力：

- A 股是 enabled market module，数据源边界是用户手工导出的本地 CSMAR 文件。
- Evaluation layer、artifact writer/loader、schema `0.7`、diagnostics framework、`manifest.display`、API、下载和 workbench 已内置到 `quant_research_platform`。
- Factor Studio / Pipeline Builder / Data Checkpoints 是平台主流程；CSV upload 只作为 legacy/dev uploaded-panel smoke / compatibility 入口保留。
- Pipeline Builder 前端默认生成配置快照，调用 `POST /api/pipeline-snapshots`；也可切换到执行模式，调用 `POST /api/pipeline-runs`。
- Factor Factory 已有后端 registry、批量评估、API 浏览入口和本地 workbench 浏览入口。
- Evidence workbench 只展示 artifact/API 结果，不重新计算 eligibility、tradability、IC、returns、weights 或 turnover。
- Crypto 和其他市场只显示为 planned / disabled，不代表真实数据、交易或空实现包。

## 核心结果

| 发现 | 证据入口 | 状态 |
|---|---|---|
| 平台合同已迁入 `quant_research_platform` 命名空间 | `EVALUATION_INTERFACE.md`、`docs/code-map/evaluation-artifacts-api.md`、代码和 tests | 已验证 |
| A 股 pipeline 已采用 snapshot/run split | `POST /api/pipeline-snapshots` 保存配置；`POST /api/pipeline-runs` 用 `start_from` 执行到 evaluation | 已验证 |
| 当前 2023-2025 evidence 可作为回归入口 | `PROJECT_STATUS.md` 记录当前有效样例；artifact 仍属于 private real-data boundary | 已验证 |
| N5/N6 和 pipeline reliability 的最新状态 | `PROJECT_STATUS.md`；N5 private smoke run id: `n5_private_smoke_2025h2_rev120_v1` | smoke passed / reliability hardening merged / N6 handoff MVS merged |

最新 current status、blocker、next step 和 root handoff 见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。下一阶段 unfinished priorities 见 [docs/architecture/next-stage-roadmap.md](docs/architecture/next-stage-roadmap.md)。

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

当前阶段：Local Research Platform。Phase 1 已完成是历史验收状态；后续 N5/N6 独立推进，不能让 roadmap 成为当前实现范围或引入 speculative infrastructure。

## Codex Task Quick Start

Codex 开始新任务时，应先读 [AGENTS.md](AGENTS.md)，判断任务模式，并按任务表面加载最小必要上下文。

普通 QRP 任务可先读 `.codex/briefs/current-qrp-status.md` 作为短 current-context primer，再决定是否需要加载 [PROJECT_STATUS.md](PROJECT_STATUS.md)、本 README 或 roadmap；它不是 `PROJECT_STATUS.md` 替代品，改变 topic / blocker / milestone / next step / handoff 时仍需查 `PROJECT_STATUS.md`。

Medium+ task 先输出 task brief、implementation plan、scope declaration、stop conditions 和 verification plan，等用户批准后再实现。涉及 artifact schema、`manifest.display`、API/download、frontend evidence、A 股 pipeline、Factor Factory 或 generated runs 时，按 `.codex/skills/quant-research-platform-maintenance/` 的 context routing 和 contracts reference 加载当前合同。

常用 Codex 任务模板见 `.codex/templates/`；普通任务优先选 fast-path / focused 模板，`/goal`、overnight 和 review-and-fix-loop 使用 `goal-overnight.md`。

Historical plans/specs 已归档到 `docs/archive/superpowers/`；它们可用于追溯历史决策，但不是当前合同真相源。当前合同以源码、测试、`manifest.display`、根目录文档、`docs/architecture/`、`docs/code-map/` 和项目维护 skill 为准。

## 当前结构

```text
src/quant_research_platform/
  evaluations/           EvaluationSpec, run_evaluation, panel contract
  kernels/               DataFrame / tensor evaluation helpers
  artifacts/             writer, loader, registry, display contract
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
PROJECT_STATUS.md       当前 operational truth、blocker、next step、handoff
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

A 股真实私有数据流程优先使用 market module 自身入口，不要为了日期过滤临时绕到通用 `evaluate-parquet`。具体命令见 [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md) 和 [docs/code-map/a-share-data-and-pipeline.md](docs/code-map/a-share-data-and-pipeline.md)。

## Artifact 和 Workbench 合同

Evaluation run 使用 artifact schema `0.7`。`src/quant_research_platform/artifacts/schema_registry.py` 与 tests 是字段数、grain、unique key、required_if / nullable_if、sentinel、generated files、display/download eligibility 的 source of truth。

`manifest.display` 是 API、下载、CSV preview、React 表格、图表和“全选下载”的共同展示合同。本地 API 支持多个 run root；前端和下载接口中的 run 身份是 `(source_alias, run_id)`，不能只按 `run_id` 折叠或缓存。

Artifact / schema / workbench 细节见 [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md)、[ARCHITECTURE.md](ARCHITECTURE.md) 和 [docs/code-map/evaluation-artifacts-api.md](docs/code-map/evaluation-artifacts-api.md)。当前有效样例和 private smoke 状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。

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
- N5 full risk/regime/cost/capacity capability 与 N6 后续扩展仍需单独设计和验证；当前 N5 v1 只代表 descriptive proxy diagnostics 和 artifact/display contract groundwork。
- N6 local baseline MVS 只代表 local research baseline / prediction handoff：记录 OOS / embargo / fold metadata，并让 evaluation 消费 prediction；private-output hardening 只加强本地边界，不是 alpha proof、真实风险模型、真实容量模型、实盘或投资建议。
- 模拟盘不属于当前范围；未来如果做，必须保持与真实交易接口隔离，并单独审批。
- CSMAR 明细数据和可还原 vendor 数据的报告不公开。

## 阅读顺序

1. [README.md](README.md)：项目公开入口。
2. [PROJECT_STATUS.md](PROJECT_STATUS.md)：当前 operational truth。
3. [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md)：evaluation 输入输出合同。
4. [ARCHITECTURE.md](ARCHITECTURE.md)：平台架构。
5. [docs/code-map/README.md](docs/code-map/README.md)：源码阅读地图。
6. [docs/architecture/next-stage-roadmap.md](docs/architecture/next-stage-roadmap.md)：unfinished priorities。
7. [REPORT.md](REPORT.md)：完整研究叙事。
8. [RESEARCH_CARD.md](RESEARCH_CARD.md)：证据底稿。
9. [AGENTS.md](AGENTS.md)：Codex 规则。
