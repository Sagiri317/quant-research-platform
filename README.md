# Quant Research Platform

本项目是本地多市场量化研究平台。当前第一版完整 market module 是 A 股：它把用户已批准的本地 CSMAR 数据转换为可复现的 research panel、factor table、pipeline snapshot 和 evaluation artifacts，并在同一项目内提供 evaluation layer、artifact contract、FastAPI evidence API 和 React evidence workbench。

本项目仍保留严格边界：它不是投资建议、实盘交易、paper trading、生产交易系统或部署服务；不登录 CSMAR，不自动下载数据，不连接券商、交易所、外部行情 API，不处理账号、密码、cookie、token 或 `.env`。

## 平台定位

```text
local approved data
-> market module
-> research panel
-> factor / label / prediction inputs
-> evaluation layer
-> schema 0.5 artifacts
-> API / downloads / evidence workbench
-> report and research evidence
```

当前能力：

- A 股是第一版完整 market module，数据源边界是用户手工导出的本地 CSMAR 文件。
- evaluation layer、artifact writer / loader、schema `0.5`、`manifest.display`、API、下载和 workbench 已内置到 `quant_research_platform`。
- Factor Studio / Pipeline Builder / Data Checkpoints 是平台主流程；CSV upload 只作为辅助输入方式保留。
- Pipeline Builder 当前只生成因子表达式/配置快照，不执行完整 A 股 build/evaluate 编排；一键全流程需要后续接入执行层。
- crypto 和其他市场只显示为 planned / disabled，不代表已支持真实数据或交易，也不保留空实现包。

## 当前结构

Phase 1 只落地本地研究平台结构，不创建 `apps/api`，也不保留空 `markets/crypto/` 实现。

```text
src/quant_research_platform/
  evaluations/           EvaluationSpec、run_evaluation 和 panel contract
  kernels/              DataFrame / tensor evaluation helpers
  artifacts/            schema 0.5 artifact writer、loader、display contract
  interfaces/
    api/                local FastAPI evidence API、downloads、run repository
    cli/                qrp command entry
  datasets/             generic CSV / Parquet readers
  markets/
    implementations/
      a_share/          CSMAR-backed A-share market module
  pipelines/
    a_share/            A-share workflow runner and checkpoints
  research/             Factor Factory and future research contracts

web/                    React/Vite evidence workbench
config/a_share/         A-share factor templates and pipeline configs
scripts/local/          本机 API / workbench 启动脚本
scripts/build/          前端打包和全量验证脚本
data/                   local private data boundary
reports/private_real/   local private real-data outputs
docs/code-map/          source reading map
```

## 运行

常用验证：

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests
```

CLI 入口：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.interfaces.cli --help
conda run -n analysis python -m quant_research_platform.interfaces.cli a-share --help
conda run -n analysis python -m quant_research_platform.interfaces.cli evaluate-parquet --help
```

## A 股专用接口

A 股真实私有数据流程优先使用 market module 自己的入口，不要为了日期过滤临时绕到通用 `evaluate-parquet`。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"

# raw/interim 已存在时，按年份生成 full research panel
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.datasets.build_panel `
  --start-year 2014 `
  --end-year 2025 `
  --end-date 2025-12-31

# 从 full panel 生成 slim factor/label Parquet
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.factors.build_factors `
  --start-year 2014 `
  --end-year 2025 `
  --windows 20,60,120 `
  --horizons 1,5,10,20,60,120

# 用 A 股 adapter 评估；start-year/end-year 定位覆盖文件，start-date/end-date 做行过滤
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

`start_year/end_year` 用于寻找 `data/processed_private/csmar/price_volume_factors_full/price_volume_factors_<start>_<end>.parquet`，如果精确文件不存在，会选择能覆盖该年份区间的更宽文件。`start_date/end_date` 是评估阶段的 `trade_date` 过滤。通用 `evaluate-parquet` 仅用于非 A 股 adapter 的本地 Parquet 验收或外部输入，不是 A 股全流程主入口。

前端验证在 `web/` 下运行：

```powershell
npm test -- --run --maxWorkers=1
npm run lint
npm run build
```

也可以使用根目录脚本：

```powershell
.\scripts\build\check_all.ps1
.\scripts\build\build_web.ps1
```

本机查看 2023-2025 私有 smoke run：

```powershell
.\scripts\local\start_private_workbench.ps1
.\scripts\local\stop_workbench.ps1
```

启动脚本会把 API 的 `QRP_RUN_ROOTS` 指到 `reports/private_real/factor_evaluation/` 下的 2023-2025 `tensor` / `dataframe`、`long_only` / `long_short` 四组 evaluation runs。它只使用相对路径，不提交或复制私有结果；如果端口被其他项目占用，脚本会停止并要求换端口或先处理占用进程。

## 私有数据边界

以下目录不提交、不公开：

```text
data/raw_private/
data/interim_private/
data/processed_private/
reports/private_real/
```

CSMAR raw/interim/processed 数据、可还原 vendor 明细、私有真实报告和任何凭据都不得复制到公开材料、GitHub、issue、PR 或聊天记录中。公开材料只允许使用代码、方法说明和脱敏聚合结论，且必须避免写成真实收益承诺或投资建议。

## Artifact 和 Workbench 合同

平台输出的 evaluation run 使用 artifact schema `0.5`。`manifest.display` 是 API、下载、CSV preview、React 表格、图表和“全选下载”的共同展示合同。

本地 API 支持多个 run root；前端和下载接口中的 run 身份是 `(source_alias, run_id)`，不能只按 `run_id` 折叠或缓存。

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
