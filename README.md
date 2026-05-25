# Quant Research Platform

## N7 当前 Evaluation 合同（2026-05-25）

当前新生成的 evaluation artifacts 使用 schema `0.6`。一个 API-visible
evaluation run 可以包含多个 `combination_id`，组合维度由
`dimensions/combinations.csv` 管理；第一阶段 required columns 仅为
`combination_id, feature, label, kernel, portfolio_mode`。成本、权重方法、
benchmark、universe、eligibility 和 tradability 等 run-level constant 配置写入
`config.json` / `lineage.json`，不写入 combinations 维度表。

Schema `0.6` 的核心输出包括 `portfolio/weights.parquet`、
`portfolio/cumulative_returns.csv`、`metrics/performance_summary.csv`、
`diagnostics/tradability_diagnostics.csv`、`input_quality/*.csv` 和
`summaries/evaluation_summary.csv`。公开合同使用 `kernel` 和
`return_source_column`；不再使用旧的 external `engine` / `pnl_return_column`
命名。A 股第一阶段只接受显式 `--features`，并使用一行多值
`--features / --labels / --kernels / --portfolio-modes`。

`--input-file` 只保留给后端和本地 CLI；API request models、frontend forms、
Pipeline Builder payloads 和 saved snapshots 不暴露该字段。`--dry-run` 会检查
输入文件、日期覆盖、required columns、组合展开和输出目录，不写 artifact。

已完成 2023H1 private smoke：
`reports/private_real/factor_evaluation/n7_eval_2023h1_smoke_v1`，覆盖
`3 features x 2 labels x 2 kernels x 2 portfolio modes = 24` 个 combinations。
完整 2023-2025 canonical regeneration 尚未在本轮提前执行。

本项目是本地多市场量化研究平台。当前第一个完整 market module 是 A 股：它把用户已批准的本地 CSMAR 数据转换为可复现的 interim 数据、research panel、factor table、pipeline snapshots 和 evaluation artifacts，并在同一项目内提供 evaluation layer、artifact contract、FastAPI evidence API 和 React evidence workbench。

边界不变：本项目不是投资建议、实盘交易、paper trading、生产交易系统或部署服务；不登录 CSMAR，不自动下载数据，不连接券商、交易所或外部行情 API，不处理账号、密码、cookie、token 或 `.env`。

## 平台定位

```text
local approved data
-> market module
-> research panel / factor table
-> evaluation layer
-> schema 0.6 artifacts
-> API / downloads / evidence workbench
-> report and research evidence
```

当前能力：

- A 股是 enabled market module，数据源边界是用户手工导出的本地 CSMAR 文件。
- Evaluation layer、artifact writer/loader、schema `0.6`、`manifest.display`、API、下载和 workbench 已内置到 `quant_research_platform`。
- Factor Studio / Pipeline Builder / Data Checkpoints 是平台主流程；CSV upload 只作为辅助入口保留。
- Pipeline Builder 前端默认生成配置快照，调用 `POST /api/pipeline-snapshots`；也可切换到执行模式，调用 `POST /api/pipeline-runs`。
- 可执行 pipeline 调用的公开控制参数是 `kernel` 和 `start_from`，支持 `existing_factors`、`rebuild_factors`、`rebuild_panel`、`rebuild_interim`，并且一定运行到 evaluation。
- Factor Factory 已有后端 registry、批量评估、API 浏览入口和本地 workbench 浏览入口；candidate summary 按 `candidate_id + combination_id` 展开 N7 evaluation results。
- Crypto 和其他市场只显示为 planned / disabled，不代表真实数据、交易或空实现包。

## 当前结构

```text
src/quant_research_platform/
  evaluations/           EvaluationSpec, run_evaluation, panel contract
  kernels/               DataFrame / tensor evaluation helpers
  artifacts/             schema 0.6 writer, loader, display contract
  interfaces/api/        local FastAPI evidence API, downloads, run repository
  interfaces/cli/        qrp command entry
  datasets/              generic CSV / Parquet readers
  markets/implementations/a_share/
                         CSMAR-backed A-share market module
  pipelines/a_share/     snapshots, start_from runner, checkpoints
  research/              Factor Factory and future research contracts

web/                     React/Vite evidence workbench
config/a_share/          A-share factor templates and pipeline configs
scripts/                 local launch, acceptance, and check scripts
data/                    local private data boundary
reports/private_real/    local private real-data outputs
docs/code-map/           source reading map
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

## 常用验证

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests
```

前端验证在 `web/` 下运行：

```powershell
npm test -- --run --maxWorkers=1
npm run lint
npm run build
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
  --features mom_20d,vol_20d,amihud_20d `
  --labels forward_return_5d,forward_return_20d `
  --portfolio-modes long_only `
  --kernels dataframe `
  --report-root reports/private_real/factor_evaluation/example_2024_dataframe_long_only
```

## 私有数据边界

以下目录不提交、不公开：

```text
data/raw_private/
data/interim_private/
data/processed_private/
reports/private_real/
```

CSMAR raw/interim/processed 数据、可还原 vendor 明细、私有真实报告和任何凭据都不得复制到公开材料、GitHub、issue、PR 或聊天记录中。公开材料只允许使用代码、方法说明和脱敏聚合结论，并且必须避免写成真实收益承诺或投资建议。

## Artifact 和 Workbench 合同

Evaluation run 使用 artifact schema `0.6`。`manifest.display` 是 API、下载、CSV preview、React 表格、图表和“全选下载”的共同展示合同。

本地 API 支持多个 run root；前端和下载接口中的 run 身份是 `(source_alias, run_id)`，不能只按 `run_id` 折叠或缓存。

Phase 1 验收结论见 [docs/architecture/phase-1-acceptance-record.md](docs/architecture/phase-1-acceptance-record.md)。2026-05-23 已完成代码复核和测试验收；Phase 1 功能已补齐：

- Factor Factory 本地 workbench 浏览入口；
- A 股 existing-factors start layer 小样本示例和 old/new 比较；
- Factor Factory 小样本示例。

本项目内已记录 root handoff signal；workspace-root 状态回传属于工作区级同步任务。Phase 1 之后的项目内架构债见 [docs/architecture/next-stage-roadmap.md](docs/architecture/next-stage-roadmap.md)。当前 A 股 pipeline 对外合同是 snapshot vs run：snapshot 只保存配置，run 只用 `start_from` 决定开始层，并固定运行到 evaluation。当前 ZIP packaging 规则已下沉到 `artifacts/`；下一批优先级仍包括前端 feature split、eligibility / tradability 诊断可见化、risk/regime/cost/capacity 和 ML prediction handoff。

新增或修改 artifacts 时必须同时检查：

- writer / loader 是否仍只接受当前 schema；
- `manifest.display.tables`、downloads、presets、charts 是否覆盖实际展示项；
- API evidence、direct download、ZIP download 是否只暴露 display contract 中的条目；
- 前端 run switching / combination switching 是否不会显示旧状态；
- docs 和 code map 是否同步。

## 阅读顺序

1. [PROJECT_STATUS.md](PROJECT_STATUS.md)
2. [ARCHITECTURE.md](ARCHITECTURE.md)
3. [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md)
4. [docs/code-map/README.md](docs/code-map/README.md)
5. [docs/csmar_data_manual.md](docs/csmar_data_manual.md)
6. [RESEARCH_CARD.md](RESEARCH_CARD.md)
7. [REPORT.md](REPORT.md)
