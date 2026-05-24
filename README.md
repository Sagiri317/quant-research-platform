# Quant Research Platform

本项目是本地多市场量化研究平台。当前第一个完整 market module 是 A 股：它把用户已批准的本地 CSMAR 数据转换为可复现的 interim 数据、research panel、factor table、pipeline snapshots 和 evaluation artifacts，并在同一项目内提供 evaluation layer、artifact contract、FastAPI evidence API 和 React evidence workbench。

边界不变：本项目不是投资建议、实盘交易、paper trading、生产交易系统或部署服务；不登录 CSMAR，不自动下载数据，不连接券商、交易所或外部行情 API，不处理账号、密码、cookie、token 或 `.env`。

## 平台定位

```text
local approved data
-> market module
-> research panel / factor table
-> evaluation layer
-> schema 0.5 artifacts
-> API / downloads / evidence workbench
-> report and research evidence
```

当前能力：

- A 股是 enabled market module，数据源边界是用户手工导出的本地 CSMAR 文件。
- Evaluation layer、artifact writer/loader、schema `0.5`、`manifest.display`、API、下载和 workbench 已内置到 `quant_research_platform`。
- Factor Studio / Pipeline Builder / Data Checkpoints 是平台主流程；CSV upload 只作为辅助入口保留。
- Pipeline Builder 前端默认只生成配置快照，调用 `POST /api/pipeline-snapshots`。
- 可执行 pipeline 调用 `POST /api/pipeline-runs`，公开控制参数是 `start_from`，支持 `existing_factors`、`rebuild_factors`、`rebuild_panel`、`rebuild_interim`，并且一定运行到 evaluation。
- Factor Factory 已有后端 registry、批量评估、API 浏览入口和本地 workbench 浏览入口。
- Crypto 和其他市场只显示为 planned / disabled，不代表真实数据、交易或空实现包。

## 当前结构

```text
src/quant_research_platform/
  evaluations/           EvaluationSpec, run_evaluation, panel contract
  kernels/               DataFrame / tensor evaluation helpers
  artifacts/             schema 0.5 writer, loader, display contract
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
  --factors mom_20d,vol_20d,amihud_20d `
  --labels forward_return_5d,forward_return_20d `
  --portfolio-mode long_only `
  --engine dataframe `
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

Evaluation run 使用 artifact schema `0.5`。`manifest.display` 是 API、下载、CSV preview、React 表格、图表和“全选下载”的共同展示合同。

本地 API 支持多个 run root；前端和下载接口中的 run 身份是 `(source_alias, run_id)`，不能只按 `run_id` 折叠或缓存。

## 阅读顺序

1. [PROJECT_STATUS.md](PROJECT_STATUS.md)
2. [ARCHITECTURE.md](ARCHITECTURE.md)
3. [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md)
4. [docs/code-map/README.md](docs/code-map/README.md)
5. [docs/csmar_data_manual.md](docs/csmar_data_manual.md)
6. [RESEARCH_CARD.md](RESEARCH_CARD.md)
7. [REPORT.md](REPORT.md)
