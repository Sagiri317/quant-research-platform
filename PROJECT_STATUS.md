# Quant Research Platform 项目状态

本文只记录当前 operational truth：当前状态、当前有效样例、当前 blocker / next steps、
以及需要同步到 workspace root 的 handoff 信号。历史 closeout、完成分支细节和旧
schema 验收故事不在本文展开；Phase 1 历史验收见
`docs/architecture/phase-1-acceptance-record.md`，下一阶段任务见
`docs/architecture/next-stage-roadmap.md`，实现流向见 `docs/code-map/`。

## Current Root-Synced Signal / 当前已同步信号

Schema `0.7` / N5 MVS descriptive proxy diagnostics 已 merge to `main`，并已完成
N5 private real-data smoke。Smoke run id 为
`n5_private_smoke_2025h2_rev120_v1`；数据范围只记录为 2025H2 aggregate
smoke，不记录 row-level private data、股票样本或完整 private paths。当前
evaluation-set artifacts 使用 `run_id`、`summary.json`、`run_context.json`、
registry-backed required tables、`manifest.display` compact download refs、
generic `/tables/{table_key}/preview` 和
`src/quant_research_platform/evaluations/diagnostics/` 下的 post-run diagnostics。

已落地范围包括 return-path risk proxy、flat bps cost sensitivity via
`scenario_metrics`、liquidity/capacity descriptive proxy、market-cap/size
descriptive proxy、missing-input-aware diagnostic status，以及 diagnostics-enabled
A-share month breakdown。它们仍只是 descriptive proxy diagnostics，不是 alpha
proof、validated risk model、Barra exposure、real capacity analysis、simulation、
paper/live trading、order routing、deployment 或 investment advice。Full backend
pytest 和 ruff 已通过。Private CSMAR data 和 `reports/private_real/` 继续保持本地私有，
不进入 portable metadata。N5 private smoke 已验证 schema `0.7` validation
passed；`diagnostic_summary`、`diagnostic_metrics`、`scenario_metrics`、
`exposure_capacity_summary` 和 `portfolio_risk_summary` 已检查；
`manifest.display`、API preview 和 selected ZIP smoke passed。Workbench browser
acceptance 本轮未运行；API/display evidence 足以证明 generic diagnostics table
visibility。

Pipeline failed-run reliability MVS 已进入 `main`。已完成范围包括 persisted
`runtime_summary.json` status fallback、failed-state checkpoints consistency、
pipeline-local stable `failure_code`、sanitized failed status/checkpoints payload，
以及 failed pipeline partial evaluation 不暴露为 completed evidence run。
该 MVS 不包含 retry/resume/overwrite、global API error envelope migration、
warnings system expansion、schema change、`manifest.display` change、frontend
change、N5/N6 expansion 或 scheduler/worker/deployment。

N6 prediction handoff guardrails、synthetic handoff smoke 和 local numpy/pandas
ML baseline trainer 已进入 `main`。这些工作只建立初始 handoff / smoke / local
baseline 入口，不代表完整 ML pipeline、deep learning backend、model registry、
feature store、A-share prediction adapter、真实风险/容量模型、交易仿真、
paper/live trading、deployment、scheduler、worker 或 external connector。

Prompt-template sync / prompt-template polish 已进入 `main`。repo 内短模板当前支持
Expanded Prompt Mode、Repo Router Mode、`auto-task-router`、closed-loop execution、
blocker / escalation gate、candidate-based Documentation Impact Gate、
direct-main docs-only、single-branch cleanup、post-merge documentation sync 和
dependency / contract approval。

Root handoff status: closed 2026-06-01. Workspace-root planning/status 已同步
N5 MVS、N5 private real-data smoke、pipeline failed-run reliability MVS、N6 initial
handoff / smoke / local baseline，以及 prompt-template sync / polish 状态；下一步
转为 N6 walk-forward / OOS / embargo contract focused audit，或先做 N6 ML baseline
interface / CLI decision audit。不要直接进入 broad implementation。

## 当前状态

| 项目 | 当前事实 |
|---|---|
| 项目定位 | 本地多市场量化研究平台；A 股是第一版完整 market module |
| 当前阶段 | Local Research Platform；Stable Core + High-Velocity Research Edge；contract-first / evidence-flow-first |
| 当前分支 | `main`；schema `0.7` / N5 MVS descriptive proxy diagnostics 已 merge to `main`；N5 private real-data smoke passed；pipeline failed-run reliability MVS 已 merge to `main`；N6 prediction handoff guardrails / synthetic smoke / local ML baseline trainer 已进入 `main`；prompt-template sync / polish 已进入 `main`；下一步是 N6 walk-forward / OOS / embargo contract focused audit，或先做 N6 ML baseline interface / CLI decision audit |
| Conda 环境 | `analysis` |
| 代码结构 | 主包为 `src/quant_research_platform/`；当前核心目录包括 `interfaces/api`、`interfaces/cli`、`datasets`、`markets/implementations/a_share`、`pipelines/a_share`、`research/factor_factory` |
| 平台合同 | evaluation contract、artifact schema `0.7`、diagnostics registry、N5 MVS descriptive proxy diagnostics、`manifest.display`、FastAPI evidence API、下载和 React evidence workbench 已迁入平台命名空间；portable metadata 使用 `run_context.json` |
| A 股 Pipeline | 执行入口为 `POST /api/pipeline-runs`；公开参数是 `start_from`，支持 `existing_factors`、`rebuild_factors`、`rebuild_panel`、`rebuild_interim`，并固定运行到 evaluation；failed-run reliability MVS 已补齐 persisted status fallback、failed checkpoints consistency、stable `failure_code`、sanitized failed payload 和 partial failed evaluation guard |
| Factor Factory | 已有 registry、批量评估、artifact `0.1`、API 和 workbench 浏览入口；候选评估通过 existing-factors start layer 调用统一 runner，并用 `candidate_id + combination_id` 汇总 evaluation results |
| 市场范围 | A 股 enabled；crypto / other markets planned / disabled，不代表真实支持，也不保留空实现包 |
| 私有边界 | CSMAR raw/interim/processed 和 `reports/private_real/` 不提交、不公开 |
| 交易边界 | 不做投资建议、paper trading、live trading、券商/交易所/API 连接或部署服务 |

## 当前有效样例

以下样例是当前可用的本地私有 evidence / regression anchors，仍属于 private
real-data outputs，不提交、不公开、不复制到聊天或公共材料中：

- `reports/private_real/factor_evaluation/schema07_eval_2023_2025_v1`：当前
  2023-2025 schema `0.7` canonical sample，含 `rev_120d`。
- `reports/private_real/factor_evaluation/schema07_eval_2023_2025_rev120_v1`：
  当前 2023-2025 schema `0.7` `rev_120d` kernel / portfolio mode parity shard。
- `reports/private_real/ff/20260527_rev120_factory_smoke_v1`：当前 Factor
  Factory `rev_120d` smoke 样例。
- N5 private real-data smoke 已通过：run id
  `n5_private_smoke_2025h2_rev120_v1`，数据范围只记录为 2025H2 aggregate
  smoke；schema `0.7` validation、diagnostics tables、`manifest.display`、API
  preview 和 selected ZIP smoke 均通过。
- `data/processed_private/csmar/panel_full/research_panel_2014_2025.parquet`：
  full research panel。
- `data/processed_private/csmar/price_volume_factors_full/price_volume_factors_2014_2025.parquet`：
  slim factor and label table。

Historical schema `0.5` / `0.6` runs, old 2023H1 smoke, N7 `n7_eval_*` runs,
and `reports/private_real/factor_evaluation/full_2024_2025/` are historical
evidence only; they are not current valid run-contract examples.

## 当前 blocker / 风险

1. Full N5 risk/regime/cost/capacity capability 尚未设计完成；当前 N5 MVS 只能作为
   descriptive proxy diagnostics groundwork，不代表真实风险模型或容量模型。
2. N6 ML prediction handoff 仍需 focused audit / interface decision；当前 baseline 不代表完整 ML pipeline 或 A-share prediction adapter，不要混入 N5 full diagnostics 分支。
3. Pipeline retry/resume/overwrite 仍未实现；global API error envelope 也仍未迁移。
4. 单个 32-combination 2023-2025 full run 仍受 weights/remap memory path 约束；扩展前
   应先设计低内存写出路径，或继续按 shard 作为正式回归输入。
5. `tradability_diagnostics_summary` 暂缓；若需要全交易日平均诊断，应新增
   artifact/API summary，而不是从 preview 前几行临时计算。

## 下一步

1. 先做 N6 walk-forward / OOS / embargo contract focused audit；或先做 N6 ML baseline interface / CLI decision audit。
2. 单独设计 N6 ML prediction handoff；保持 prediction input contract 与现有 factor
   evaluation contract 分离。
3. 基于当前 2023-2025 canonical / `rev_120d` evidence，保留低内存 comparison、图表分页预览和浏览器验收作为后续回归门槛。
4. 如需单个 32-combination 2023-2025 run，先处理 weights/remap 低内存写出问题。

## 已完成工作摘要

- Phase 1 已完成并由 `docs/architecture/phase-1-acceptance-record.md` 作为历史验收记录保存。
- N1/N2/N3/N4/N7/N8 已合入 `main`；详细实现和旧 schema 验收故事不再作为当前状态展开，必要时查 Git history、`docs/archive/superpowers/` historical execution records 和相关 code-map。
- 2026-05-28 maintenance surface cleanup 已收敛当前脚本/agent helper 边界：用户可见 workbench scripts 保留在 `scripts/`，agent-only verification helpers 归项目维护 skill，`docs/archive/superpowers/` plans/specs 作为 historical execution records。

## Root Handoff / 根目录回传信号

子项目会话如果改变当前主题、阶段、里程碑、阻塞、下一步、公开成熟度，或
数据/API/Git/Conda/隐私边界，应在本表新增 `open` 信号。根目录同步完成后，
将状态改为 `closed` 并记录同步日期。普通实现细节、测试、notebook 调试和
局部代码变更不进入根目录，除非它们改变里程碑级状态。

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-06-01 | status | N6 prediction handoff guardrails、synthetic handoff smoke、local numpy/pandas ML baseline trainer 已进入 `main`；prompt-template sync / polish 已进入 `main`。当前仍不是完整 ML pipeline、deep learning backend、model registry、feature store、A-share prediction adapter、paper/live trading、deployment、scheduler、worker 或 external connector。下一步改为 N6 walk-forward / OOS / embargo contract focused audit，或先做 N6 ML baseline interface / CLI decision audit；不要直接进入 broad implementation。 | WORKSPACE_STATUS, PLAN_INDEX | closed 2026-06-01 |
