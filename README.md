# Quant Research Platform

本项目是本地多市场量化研究平台。当前第一版完整 market module 是 A 股：它把用户已批准的本地 CSMAR 数据转换为可复现的 research panel、factor table、pipeline snapshot 和 evaluation artifacts，并在同一项目内提供 core evaluation、artifact contract、FastAPI evidence API 和 React evidence workbench。

本项目仍保留严格边界：它不是投资建议、实盘交易、paper trading、生产交易系统或部署服务；不登录 CSMAR，不自动下载数据，不连接券商、交易所、外部行情 API，不处理账号、密码、cookie、token 或 `.env`。

## 平台定位

```text
local approved data
-> market module
-> research panel
-> factor / label / prediction inputs
-> core evaluation
-> schema 0.5 artifacts
-> API / downloads / evidence workbench
-> report and research evidence
```

当前能力：

- A 股是第一版完整 market module，数据源边界是用户手工导出的本地 CSMAR 文件。
- core evaluation、artifact writer / loader、schema `0.5`、`manifest.display`、API、下载和 workbench 已内置到 `quant_research_platform`。
- Factor Studio / Pipeline Builder / Data Checkpoints 是平台主流程；CSV upload 只作为辅助输入方式保留。
- crypto 和其他市场只保留接口预留，不代表已支持真实数据或交易。

## 当前结构

```text
src/quant_research_platform/
  core/                 EvaluationSpec、run_evaluation 和 panel contract
  engines/              DataFrame / tensor evaluation helpers
  artifacts/            schema 0.5 artifact writer、loader、display contract
  api/                  local FastAPI evidence API、downloads、run repository
  cli/                  qrp command entry
  io/                   explicit local CSV / Parquet input helpers
  markets/
    a_share/            CSMAR-backed A-share market module
    crypto/             reserved placeholder only
  research/             future shared research contracts

web/                    React/Vite evidence workbench
config/a_share/         A-share factor templates and pipeline configs
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
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.cli --help
conda run -n analysis python -m quant_research_platform.cli a-share --help
conda run -n analysis python -m quant_research_platform.cli evaluate-parquet --help
```

前端验证在 `web/` 下运行：

```powershell
npm test -- --run --maxWorkers=1
npm run lint
npm run build
```

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
