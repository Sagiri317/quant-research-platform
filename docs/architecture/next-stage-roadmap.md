# Next Stage Roadmap / 下一阶段路线图

本文只记录当前 active roadmap、unfinished work 和 current priorities。已完成任务
只保留短状态；历史 implementation details、closeout、旧 schema 验收故事不在本文展开。

本文不引入部署、外部 API、券商连接、paper trading、live trading 或实盘能力。
最终能力边界见 [target-architecture-and-capabilities.md](target-architecture-and-capabilities.md)。

## 当前优先级

| ID | 任务 | 优先级 | 状态 | 目标 |
|---|---|---:|---|---|
| N5 | 风险、regime、成本和容量扩展 | P1 | MVS merged to `main`; N5 private real-data smoke passed; failed-run reliability MVS merged | N5 MVS descriptive proxy diagnostics 已落地并完成 2025H2 aggregate smoke；full capability 仍需单独设计和验证 |
| N6 | ML prediction handoff | P2 | initial guardrails / synthetic smoke / local baseline merged; audit next | 建立 prediction output 到 evaluation 的可复现接口 |
| Pipeline reliability second slice | Pipeline reliability | P2 | deferred behind N6 focused audits | failed-run reliability MVS 已合入；如需继续第二个 minimal vertical slice，先单独做决策审计 |
| Pipeline history UI | Pipeline run history / visibility | P3 | planned | 在不改变 pipeline 语义的前提下改善本地 workbench 对 run history 的读取和展示 |

当前推荐顺序：

```text
1. N6 walk-forward / OOS / embargo contract focused audit
2. N6 ML baseline interface / CLI decision audit if needed
3. Pipeline history UI
```

当前基线：

- N1/N2/N3/N4/N7/N8 已合入 `main`，后续扩展应基于 schema `0.7`、
  `combination_id`、portable display preview metadata 和 artifact/API/frontend 合同。
- N5 当前不再是纯 planned：schema `0.7` 已落地 N5 MVS descriptive proxy diagnostics
  并 merge to `main`；N5 private real-data smoke 已通过，run id
  `n5_private_smoke_2025h2_rev120_v1`，数据范围仅记录为 2025H2 aggregate
  smoke。Pipeline failed-run reliability MVS 已进入 `main`，覆盖 status fallback、
  failed-state consistency、stable `failure_code`、sanitization 和 partial failed
  evaluation guard；retry/resume/overwrite 与 global API error envelope migration
  仍未实现。本文中的 full N5 capability 仍指后续 full risk/regime/cost/capacity capability。
- N6 prediction handoff guardrails、synthetic handoff smoke 和 local numpy/pandas
  ML baseline trainer 已进入 `main`。这不是完整 ML pipeline、deep learning backend、
  model registry、feature store、A-share prediction adapter、paper/live trading、
  deployment、scheduler、worker 或 external connector。
- Prompt-template sync / polish 已进入 `main`；repo 内短模板支持 Expanded Prompt Mode、
  Repo Router Mode、`auto-task-router`、closed-loop execution、blocker / escalation
  gate、candidate-based Documentation Impact Gate、direct-main docs-only、
  single-branch cleanup、post-merge documentation sync 和 dependency / contract
  approval。
- 2023-2025 canonical sample、`rev_120d` Factor Factory smoke、低内存 comparison 和
  frontend acceptance 已生成；后续不要直接扩大一次性 full run。

## 已完成任务短状态

| ID | 状态 | 当前意义 |
|---|---|---|
| N1 A 股 start-from pipeline runner | merged to `main` | 当前 pipeline API 为 snapshot/run split；executable run 使用 `start_from` 并固定运行到 evaluation |
| N2 Eligibility / tradability 诊断可视化 | merged to `main` | 当前 diagnostics 通过 artifact/API/display/workbench 消费；如需全期间 summary，应另行设计 `tradability_diagnostics_summary` |
| N3 Artifact ZIP packaging 下沉 | merged to `main` | ZIP item validation、cache identity 和 file collection 已归 `artifacts/` |
| N4 前端 feature split | merged to `main` | `web/src` 已拆为 `shell/`、`features/` 和 `shared/` |
| N7 A 股 evaluation run identity 与 2023-2025 样例生成 | merged to `main` | 当前 API-visible run 使用 `combination_id` 管理 result slices；当前有效样例见 `PROJECT_STATUS.md` |
| N8 架构简化与合同消费收敛 | merged to `main` | display preview/download helper、factor-template API、A-share service entry 和 legacy/dev upload 边界已收敛 |
| Pipeline failed-run reliability MVS | merged to `main` | A-share failed runs 已具备 persisted status fallback、failed checkpoints consistency、stable `failure_code`、sanitized failed payload 和 partial failed evaluation guard；不代表 retry/resume/overwrite 或 production reliability |

完成任务的详细 implementation notes 和历史验收记录应查 Git history、
`docs/archive/superpowers/` historical execution records、`PROJECT_STATUS.md` 的短摘要和
`docs/code-map/` 当前实现图谱；本文不再承载 closeout。

## N5 full 风险、Regime、成本和容量扩展

当前实现状态：

- N5 v1 已是 partial implementation + contract groundwork：schema `0.7` 已包含
  diagnostics registry、required diagnostics tables、conditional materialized views、
  `manifest.display`、API preview/download 和 frontend evidence flow。
- 当前 risk/regime/cost/capacity 输出是 post-run descriptive proxy diagnostics，不是
  正式 risk model、capacity model、regime engine、backtest、simulation 或 trading system。
- 已落地 MVS 包括 return-path risk proxy、flat bps cost sensitivity via
  `scenario_metrics`、liquidity/capacity descriptive proxy、market-cap/size
  descriptive proxy、missing-input-aware diagnostic status 和 diagnostics-enabled
  A-share month breakdown；full backend pytest 和 ruff 已通过。
- N5 private real-data smoke 已通过：schema `0.7` validation passed；
  `diagnostic_summary`、`diagnostic_metrics`、`scenario_metrics`、
  `exposure_capacity_summary` 和 `portfolio_risk_summary` 已检查；
  `manifest.display`、API preview 和 selected ZIP smoke passed。Workbench browser
  acceptance 未运行；API/display evidence 足以证明 generic diagnostics table
  visibility。
- 下方目标保留为 full N5 capability 的 unfinished work，不升级当前 phase，也不扩大当前实现范围。

目标：

- 把风险暴露、regime 切片、成本侵蚀、流动性/容量压力和 A 股执行假设制度化为
  evaluation artifact、API preview/download 和 workbench 可消费的诊断输出。
- 形成后续研究报告所需的稳定诊断表，而不是只停留在 notebook 或一次性脚本里。
- 让收益来源、风险暴露、成本假设、容量约束和组合构建贡献可以分开解释。

最小可交付范围：

- Signal / regime：按当时可知 regime 或事件窗口切分稳定性表。
- Risk diagnostics：market beta、行业、市值、风格、波动率、流动性暴露，以及 neutralized IC / return、drawdown、stress test 和 crowding proxy 的第一版诊断。
- Cost diagnostics：从固定 bps baseline 扩展到 flat、linear、piecewise-linear 和 quadratic cost profile。
- Capacity diagnostics：输出 liquidity bucket、ADV participation cap、capacity proxy 和 capacity curve；只解释流动性压力和容量敏感性，不写成真实可交易规模承诺。
- Portfolio construction：先补规则型权重和约束诊断，不引入优化器。
- A 股约束：显式记录涨跌停、停牌、T+1、印花税、小盘流动性、收盘成交假设，以及 `long_short` 仅为因子分组诊断口径。

非目标：

- 不写成真实收益承诺。
- 不进入交易风控、paper trading、live trading、券商/交易所连接、外部数据 API、部署服务或自动化执行能力。
- 不把 regime 用未来收益定义。
- 不训练 ML model，不混入 N6 prediction handoff。
- 不实现组合优化器、事件驱动撮合、真实成交回放、borrow availability、保证金或融券成本模型。
- 不为 N5 新增外部数据下载、凭据、API key 或自动登录。

验收：

- 新增诊断表由 artifact writer 写出，loader 和 manifest validation 能校验。
- API preview、direct download 和 ZIP download 只通过 `manifest.display` 暴露新增表。
- React evidence workbench 只消费 API/display metadata，不在前端重新计算风险、成本、容量或中性化结果。
- A 股私有样例优先复用当前 2023-2025 canonical run 与 `rev_120d` parity shard；
  不直接扩大到单个 32-combination full run，除非先处理低内存 weights/remap 写出。
- private CSMAR data、`reports/private_real/` 生成物、本机绝对路径和凭据不得进入
  Git、public-downloadable metadata、日志或普通报告。

## N6 ML Prediction Handoff

目标：

- 建立模型输出到 evaluation 的稳定接口。
- 让 prediction 像 factor 一样成为可评价输入，而不是在 evaluation layer 中训练模型。

最低字段：

- model id / version
- feature set
- label definition
- train / validation / test split
- embargo / leakage controls
- data version
- code version
- random seed
- prediction column

非目标：

- 不训练模型。
- 不把 prediction handoff 混入 N5 diagnostics。
- 不引入 model registry、feature store、deployment、paper/live trading 或外部服务。
- 当前 local baseline 不代表 alpha proof、真实风险/容量模型、交易仿真、完整 ML platform
  或 A-share prediction adapter。

验收：

- prediction Parquet 可通过 `EvaluationSpec` 或 A 股 adapter-like handoff 进入 evaluation。
- model manifest 可追踪训练数据、特征、label、split 和失败实验。
- walk-forward 和 OOS 结果明确区分。

## Pipeline history UI

目标：

- 改善本地 workbench 对 pipeline snapshot/run history 的读取和展示。
- 保持 `POST /api/pipeline-snapshots` 只保存配置、`POST /api/pipeline-runs` 执行到 evaluation 的现有语义。

非目标：

- 不新增 scheduler、worker、job system 或部署服务。
- 不改变 `start_from` 语义。
- 不把 pipeline history 变成广义实验追踪平台。

验收方向：

- 本地 UI 能清楚区分 snapshot 与 executable run。
- run history 不暴露 private filesystem paths。
- 与 evidence run selection 的关系清晰，不绕过 artifact/API/display 合同。
