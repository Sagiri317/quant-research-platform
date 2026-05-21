# Quant Research Platform 迁移设计

本文记录把当前 `a-share-alpha-lab` 改造为 `quant-research-platform` 的已确认设计。它是迁移设计，不是实施计划；真正动代码前还需要单独写任务级实施计划，并按测试门槛验收。

## 1. 目标

把当前项目升级为本地多市场量化研究平台，同时保留 `alpha-research-core` 已经做得比较好的评价引擎、artifact、API 和前端 workbench。

平台需要支持：

- 每一步中间结果都可检查；
- 也能一条命令跑完整流程；
- 生成可复查的 evaluation artifacts；
- 有本地前端研究工作台；
- A 股是第一版完整实现的 market module；
- crypto 和其他市场第一版只预留接口，不迁真实代码。

迁移完成后，新平台不应再依赖 `../alpha-research-core/src`。

## 2. 命名

已确认命名：

- 项目 / 仓库 / 目录名：`quant-research-platform`。
- Python 包名：`quant_research_platform`。
- CLI 命令：`qrp`，含义是 Quant Research Platform。
- 中文定位：本地多市场量化研究平台。
- 前端产品名：Research Evidence Workbench。

当前本地目录可以先继续叫 `a-share-alpha-lab`，避免当前 Codex 会话路径中断。等迁移和测试通过后，目录改名为 `quant-research-platform` 由用户自己执行；目录未改名不阻塞夜间迁移 goal。

`alpha-research-core` 不合并 Git 历史，只复制当前代码快照。用户已批准在旧 core 项目里写冻结 / deprecated 说明，但不删除旧 core；删除旧 core 需要迁移完成后单独审批 exact path。

## 3. 目标目录结构

目标结构：

```text
src/
  quant_research_platform/
    __init__.py
    core/
    engines/
    artifacts/
    api/
    io/
    cli/
    markets/
      __init__.py
      a_share/
        __init__.py
        csmar/
        datasets/
        factors/
        labels/
        risk/
        regime/
        evaluation/
        validation/
        runtime/
      crypto/
        __init__.py
    research/
      __init__.py
      models/
      reporting/
    runtime/
web/
tests/
  core/
  engines/
  artifacts/
  api/
  markets/
    a_share/
  web/
config/
  a_share/
data/
  raw_private/
  interim_private/
  processed_private/
reports/
  private_real/
docs/
.codex/
  skills/
```

迁移后只使用 `src/` 作为 Python 源码根目录。`pyproject.toml` 应改为只认 `pythonpath = ["src"]`，不能再包含 `../alpha-research-core/src`。

旧 `code/` 只是迁移来源。用户已批准在迁移、import 修改、测试和文档都完成后清理旧 `code/`，不保留兼容入口。清理范围只限旧源码目录 `E:/quant-workspace/projects/a-share-alpha-lab/code/`，不包括 `data/`、`reports/`、`docs/`、`notebooks/`、`config/`。

## 4. 目录审计原则

不能机械照搬目录。每个迁移过来的子目录都要回答：它在新平台里还承担什么真实职责。

core 里已经稳定的逻辑尽量少改：

- IC / RankIC；
- 分组收益；
- long-only / long-short portfolio path；
- turnover、cost、PnL、performance summary；
- artifact schema `0.5`；
- `manifest.display`；
- FastAPI evidence / download；
- React evidence 浏览、图表、CSV preview、下载、comparison。

需要重新审计的边界：

- `core/`：公开评价合同、panel / result 对象、异常。
- `engines/`：具体指标和组合公式。
- `artifacts/`：writer、loader、manifest display、inventory、comparison。
- `api/`：本地 FastAPI app、payload、download、repository、run index。
- `io/`：只保留显式 DataFrame / CSV / Parquet panel adapter；不保留隐式扫描 sibling project 或“从其他项目找 Parquet”的能力。
- `cli/`：`qrp` 命令入口。
- `markets/a_share/`：CSMAR、A 股 panel、因子、label、risk、regime、validation、A 股 evaluation adapter。
- `research/`：跨市场模型和报告约定，第一版保持轻量。
- `markets/crypto/`：第一版只做接口占位。

平台核心能力不应该依赖“读取其他本地项目的 Parquet”。market module 可以直接把内存里的 DataFrame 交给 core；CLI/API 辅助入口可以读取用户明确指定的本地 CSV 或 Parquet。

## 5. Core 评价接口

核心接口优先是：

```text
准备好的 panel DataFrame -> EvaluationSpec -> run_evaluation -> artifacts
```

文件读取只是辅助入口：

```text
明确指定的 CSV/Parquet 路径 -> DataFrame -> EvaluationSpec -> run_evaluation
```

这样 core 不需要知道数据来自 CSMAR、crypto、CSV、Parquet 还是 notebook。A 股、crypto、未来其他市场只要最后产出统一的 long panel，就能使用同一个评价接口。

保留 `EvaluationSpec` 和 `run_evaluation` 这两个概念，但迁到新命名空间：

```python
from quant_research_platform.core import EvaluationSpec, run_evaluation
```

## 6. A 股模块

当前 A 股功能从 `code/` 迁到：

```text
src/quant_research_platform/markets/a_share/
```

迁移映射：

```text
code/csmar/       -> markets/a_share/csmar/
code/datasets/    -> markets/a_share/datasets/
code/factors/     -> markets/a_share/factors/
code/evaluation/  -> markets/a_share/evaluation/
code/validation/  -> markets/a_share/validation/
code/runtime/     -> 拆到 markets/a_share/runtime/ 和 shared runtime/
```

A 股模块负责：

- CSMAR 手工导出边界；
- raw / interim / processed private data 分层；
- A 股 universe、eligible、tradability、停牌、涨跌停、ST、上市天数、流动性等 domain rules；
- factor、label、prediction、risk、regime input 生成；
- 私有报告和可公开的聚合摘要。

core 保持 market-neutral，只评价显式传入的列。

## 7. Pipeline 模型

平台既要支持分步检查，也要支持一键全流程。

分步模式示例：

```powershell
qrp a-share build-interim
qrp a-share build-panel
qrp a-share build-factors
qrp a-share evaluate
qrp serve
```

一键模式示例：

```powershell
qrp a-share run-pipeline --config config/a_share/pipelines/full_2023_2025.yaml
```

一键模式不是黑箱。内部仍然按阶段运行，并保存 checkpoint：

```text
raw/private input
-> interim audit tables
-> processed panel
-> factor/label table
-> evaluation artifacts
-> evidence workbench manifest
-> summary report
```

默认规则：

- blocking error 直接停止；
- warning 继续跑，但必须写入 run manifest、summary 和最终报告；
- 默认不覆盖已有正式输出；
- 同名 run 自动生成带时间戳的新目录；
- pipeline 默认不启动 API 或前端服务；
- 浏览结果时单独运行 `qrp serve`。

`--overwrite` 必须保守。它不能删除 raw、interim、processed 或 private report 目录。清理旧生成目录需要单独 exact-path 审批。

## 8. Blocking Error 和 Warning

Blocking error，也就是必须停止的问题：

- 必需输入文件或必需列缺失；
- 主键重复，例如 `stock_code + trade_date` 重复；
- 日期顺序或日期对齐错误；
- label 或 PnL return 列不可用；
- 明显未来数据泄漏；
- artifact schema 不支持；
- `manifest.display` 不合法；
- 未确认就要覆盖受保护结果。

Warning，也就是可以继续但必须记录的问题：

- 可选字段缺失；
- 少量 factor 值缺失；
- 某一年、某个 regime、某个 segment 或事件窗口样本量偏小；
- 行业、风险、regime 诊断暂不可用；
- 停牌、涨跌停或 tradability filter 删除了一些行；
- artifact 规模接近阈值但未超过阈值。

warning 必须能在 run detail 页面和保存的 run metadata 里看到。

## 9. 前端方向

尽量保留当前 core 前端的视觉系统、配色、组件质感、图表风格、CSV preview、下载和 evidence 浏览体验。

但产品流程从：

```text
上传 CSV -> 评价 -> 浏览 artifacts
```

改成：

```text
定义因子 / 配置 pipeline -> 运行 pipeline -> 检查 checkpoints -> 浏览 evidence
```

第一版页面：

- Research Runs：浏览已有 pipeline runs 和 evaluation runs。
- Factor Studio：选择模板，编辑受控因子表达式。
- Pipeline Builder：选择 market、日期范围、universe、成本、portfolio mode、label/PnL 列和输出选项。
- Data Checkpoints：检查 interim、panel、factor、warning、规模摘要。
- Evidence Workbench：沿用 core 的结果浏览、图表、CSV preview、下载和 comparison。

CSV upload 不再是主流程。如果保留，也只是开发或兼容入口，不作为默认入口。

第一版只有 A 股 pipeline creation 可用。crypto 和其他市场可以出现在 market selector 里，但显示 disabled / planned；后端保留接口，不接真实 runner。

## 10. Factor Studio

第一版 Factor Studio 支持受控表达式和内置模板填充，不支持任意 Python 代码执行。

允许的表达式示例：

```text
neg(return(close, 20))
zscore_cs(return(close, 60))
rank_cs(mean(volume, 20))
```

规则：

- 表达式只能使用白名单字段和白名单函数；
- 后端解析表达式 AST 或结构化表示；
- 不使用 Python `eval`；
- 不允许 `import`；
- 不允许文件、网络、shell、凭据或环境访问；
- 解析错误要给出可操作的 UI 提示；
- run config 必须保存表达式原文、解析结构、字段依赖、模板来源和参数快照。

第一版模板范围：

- 内置模板；
- 一次性自定义表达式 / 配置提交；
- 每次运行自动保存配置快照；
- 可以从旧 run 加载配置回表单。

第一版不做完整模板管理：

- 不做 save-as-template UI；
- 不做重命名、复制、删除、收藏、默认模板管理。

## 11. 配置和运行快照

采用两层保存：

1. 可复用配置模板，例如 `config/a_share/` 下的 YAML。
2. 每次 run 目录里保存当次实际配置快照。

这样旧结果可以复查，不会因为后来模板修改而说不清当时怎么跑的。

建议输出：

```text
reports/private_real/pipeline_runs/<run_id>/run_config.yaml
reports/private_real/pipeline_runs/<run_id>/warnings.json
reports/private_real/pipeline_runs/<run_id>/summary.md
reports/private_real/evaluations/<eval_run_id>/manifest.json
```

旧 run 页面读取的是 run 目录里的快照，不读取最新模板。

## 12. Artifact 和 Workbench 合同

`alpha-research-core` 当前 artifact 合同是最稳的部分，迁移到新平台后应继续保留。

规则：

- 当前 artifact schema 是 `0.5`；
- `manifest.display` 是表格、下载、图表展示的唯一合同；
- 每个 display table 都要有匹配的 download item；
- 新增用户可见 artifact 必须进入 `manifest.display`；
- 不恢复 `artifacts.xlsx`；
- 不恢复旧的 `total_*_portfolio_return` summary 字段；
- 不为了旧 generated run 加宽松 fallback，旧 run 应重新生成；
- run / combination 切换不能显示上一个 run 的 stale evidence。

任何 artifact、API、download 或 frontend evidence 变更，都必须同轮更新 producer、loader、API、前端 consumer、测试、文档和 code map。

## 13. API 方向

本地 API 继续用 FastAPI，保持 local-only。

预留 endpoint 家族：

```text
GET  /markets
GET  /markets/{market}/capabilities
GET  /factor-templates
POST /pipeline-runs
GET  /pipeline-runs/{id}/status
GET  /pipeline-runs/{id}/checkpoints
GET  /evaluation-runs
GET  /evaluation-runs/{id}/evidence
GET  /evaluation-runs/{id}/tables/{table_id}
GET  /evaluation-runs/{id}/downloads/{item_id}
```

具体实现时可以沿用或调整旧 core endpoint 风格，但 API 必须区分：

- pipeline 创建和 checkpoint 状态；
- evaluation artifact evidence 和下载；
- market capabilities 和 factor templates。

本次迁移不做 deployed service、认证系统、浏览器下载自动化、凭据处理、券商 connector、交易所 connector、模拟盘或实盘。

## 14. Skills、子智能体和执行方式

实施时必须参考：

- 全局 `AGENTS.md`；
- 当前项目 `AGENTS.md`；
- 旧 core 项目 `AGENTS.md`；
- 旧 core 项目 skill：`.codex/skills/alpha-research-core-maintenance/SKILL.md`；
- Windows/Conda、frontend design、React best practices、TDD、verification、parallel agents 等相关全局 skill。

用户已允许使用多个子智能体，包括超过 5 个，只要任务确实独立。

适合拆分的并行工作流：

- core contracts、engines、artifacts；
- API、run repository、downloads、payloads；
- web 视觉搬迁和 Factor/Pipeline Studio；
- A 股 market module 迁移；
- tests 和 import rewrite；
- docs、AGENTS、skills、PROJECT_STATUS；
- factor expression / config contract。

主线程负责分配不冲突的写入范围、审查子智能体结果、整合和最终验收。

## 15. 测试和验收门槛

迁移未达到以下条件前，不能称为完成：

```text
1. 不再依赖 ../alpha-research-core/src。
2. active code 中不再有 alpha_research_core import path；迁移说明或历史解释除外。
3. Python tests pass。
4. Ruff pass。
5. Frontend tests pass。
6. Frontend lint pass。
7. Frontend build pass。
8. A 股 evaluation 能通过新 platform core 跑通。
9. Research Evidence Workbench 能消费新平台 artifacts。
10. docs、AGENTS、skills、PROJECT_STATUS、code maps 描述一致。
```

推荐最终验证命令：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests
conda run -n analysis python -m quant_research_platform.cli --help
conda run -n analysis python -m quant_research_platform.cli a-share --help
```

前端命令从 `web/` 目录运行：

```powershell
npm test -- --run --maxWorkers=1
npm run lint
npm run build
```

如果需要浏览器验收，启动本地 API 和 Vite 服务，检查 run switching、combination switching、CSV preview、downloads、report mode、pipeline form 和至少一个窄屏视口。验收后停止服务。

资源控制规则：

- 真实 A 股 pipeline 验证先只用小样本；
- 默认使用一年数据；
- 默认只选 2-3 个因子和 2-3 个 label；
- 先跑 smoke，再考虑扩大样本；
- 不在第一轮验证中直接跑 2014-2025 全样本；
- 启动前后端前先检查是否已有服务占用端口；
- 前端 / 后端服务只在浏览器验收需要时启动；
- 验收完成后停止服务，避免长期占用内存。

## 16. 文档迁移

文档要从 A 股单项目视角改成平台视角：

- `README.md`：中文优先的平台入口。
- `AGENTS.md`：新平台的安全和执行边界。
- `PROJECT_STATUS.md`：当前迁移状态和 root handoff signal。
- `ARCHITECTURE.md`：平台模块、artifact/API 合同。
- `EVALUATION_INTERFACE.md`：market module 到 core 的合同。
- `RESEARCH_CARD.md`：证据账本和公开 / 私有边界。
- `docs/code-map/`：源码导航。
- `.codex/skills/`：基于旧 core skill 改成平台维护 skill。

当前 `docs/a_share_quant_platform_blueprint.md` 是临时前置文档。有效内容应吸收到平台文档里，然后删除这个 A 股单项目蓝图。

用户已批准删除这个临时蓝图 exact path：

```text
E:/quant-workspace/projects/a-share-alpha-lab/docs/a_share_quant_platform_blueprint.md
```

workspace root 的 planning/status 同步由用户明天再做，不阻塞夜间迁移 goal。

## 17. 隐私和数据边界

必须保留所有私有数据边界：

- CSMAR raw、interim、processed 和 private reports 保持私有；
- 不创建、读取、打印、保存或提交 credentials、`.env`、cookies、tokens 或账号绑定数据；
- 不做 CSMAR 登录、下载器自动化、API 集成或浏览器自动化；
- 本次迁移不添加外部市场 API；
- 不添加券商、交易所、模拟盘、实盘或下单功能；
- 不把研究输出写成投资建议或真实收益承诺。

可公开内容只包括代码、方法、合同和经过审查的聚合摘要。

## 18. 第一版不做的事

第一版不包含：

- 真实 crypto pipeline 迁移；
- 完整模板管理 UI；
- 任意 Python 因子代码执行；
- 部署服务或认证；
- 数据库 run storage；
- 券商 / 交易所 connector；
- 模拟盘或实盘；
- 删除旧 `alpha-research-core` 项目。

## 19. 依赖安装授权边界

用户已批准两类项目内依赖安装：

- Python 包可以安装到 Conda `analysis` 环境；
- 前端依赖可以在项目 `web/` 目录下通过 npm 安装。

执行时仍要遵守这些边界：

- 不安装到 Conda `base`；
- 不使用 `pip`，除非用户后续明确批准并接受 Conda/pip 混用风险；
- 不做全局 npm / Conda / Git 配置修改；
- 不在项目范围外写入业务文件；
- 如果 npm 或 Playwright 需要在用户缓存目录写浏览器二进制、全局 cache 或其他项目外文件，应先说明并确认；
- 如果新增依赖会改变架构或部署边界，应先写入计划再执行。

浏览器验收优先使用本机已安装 Chrome。不要默认运行 `npx playwright install chromium`；只有系统 Chrome 无法用于验收且用户再次确认时，才安装 Playwright 浏览器二进制。

## 20. 后续功能分级

`core_future_features_summary.md` 里的功能不应一次性全部塞进第一版迁移。分级如下。

第一版迁移要保留或顺手接住的能力：

- IC / RankIC；
- grouped return；
- 已有因子中性化；
- long-only；
- long-short；
- transaction cost；
- Parquet 显式输入；
- 多市场统一接口骨架；
- regime split 的接口预留；
- label / PnL return 合同；
- evidence workbench 展示、下载、comparison。

第一版可以顺手做“接口或轻量诊断”，但不做完整研究能力：

- 多因子组合：先当作显式 `composite_*` factor 列评价，不做自动组合器；
- 风险暴露分析：先支持传入 exposure columns 并展示/分组，不做 Barra-like 模型；
- liquidity / capacity：先保留字段和 summary 位置，不做容量承诺；
- Tail Risk：可以作为 performance diagnostics 的轻量指标候选，但不阻塞迁移；
- walk-forward validation：先作为模型 pipeline 后续接口，不训练模型；
- bull/bear、high-vol/low-vol、trend/mean-reversion：先按外部传入 regime column 做 breakdown，不自动检测。

必须后续单独阶段设计和实现：

- 因子正交化；
- 自动多因子组合搜索；
- 协方差矩阵；
- Barra-like 风险模型；
- 风险归因；
- beta neutral；
- optimizer；
- risk parity；
- slippage；
- market impact；
- XGBoost / LightGBM；
- ensemble model；
- regime detection；
- 多频率完整支持；
- tick 数据支持；
- 网站部署上线。

网站部署上线需要单独设计，因为它会改变当前 local-only、安全、私有数据、API 暴露、认证和部署边界。第一版迁移可以把前端工程和 build 流程整理好，但不把网站部署上线作为顺手任务。

## 21. 实施准备

本设计确认后，再写单独实施计划。实施计划要拆成小任务，每个任务可测试、可回滚、职责清楚。

实施阶段建议使用 high 或 xhigh reasoning effort，因为它同时涉及包结构、imports、指标计算、artifacts、API、React、私有数据边界、docs、tests 和项目身份。

实施计划必须明确这些节点：

- 何时复制 core 源码到新 namespace；
- 何时迁移 A 股模块到 `markets/a_share`；
- 何时更新测试；
- 何时更新前端；
- 何时更新 docs 和 skills；
- 何时删除旧 `code/`；
- 何时冻结旧 core 项目；
- 何时改本地目录名。
