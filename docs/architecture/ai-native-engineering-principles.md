# AI-Native Engineering Principles / AI 原生工程原则

> 本文是 `quant-research-platform` 的 AI-native engineering compressed mental model / short principles。它沉淀本项目在 AI-assisted engineering 实践中形成的稳定经验，用来指导人类、Codex 和其他 AI agent 共同维护一个可长期演化的量化研究平台。
>
> 本文不是任务清单，不是路线图，不是当前实现状态，也不是 Codex 的逐条 task execution 说明。具体 task-mode、recommended reasoning effort、context budget、subagent、commit/push/merge、Goal / Overnight 执行规则以 `AGENTS.md`、`.codex/skills/quant-research-platform-maintenance/SKILL.md`、`.codex/skills/quant-research-platform-maintenance/references/context-routing.md`、`.codex/templates/` 和 `.codex/briefs/current-qrp-status.md` 为准；工程合同仍以 `EVALUATION_INTERFACE.md`、`ARCHITECTURE.md`、`docs/code-map/` 和代码/测试为准。

---

## 1. 项目身份

`quant-research-platform` 不是一个单次回测脚本，也不是单一 A 股因子项目。它的真实定位是：

```text
AI-native Quant Research Platform
```

当前阶段是：

```text
Local Research Platform
```

当前核心目标是：

- factor research（因子研究）
- evaluations（评价）
- artifact generation（产物生成）
- evidence workflow（证据链工作流）
- local evidence workbench（本地证据工作台）
- A-share research reliability（A 股研究可靠性）

长期可以演化为更完整的专业量化研究基础设施，但当前阶段不等于 SaaS 平台、生产交易系统、实盘系统、paper trading 系统或分布式任务平台。

---

## 2. 总原则：AI Accelerated, Human Governed

本项目采用：

```text
AI Accelerated, Human Governed
```

AI / Codex 负责：

- implementation（实现）
- tests（测试）
- local refactor（局部重构）
- documentation sync（文档同步）
- artifact / API / frontend 同步
- factor development（因子开发）
- repetitive engineering work（重复工程工作）

人类负责：

- contracts（合同）
- architecture boundaries（架构边界）
- phase control（阶段控制）
- abstraction level（抽象层级）
- priority（优先级）
- scope approval（范围批准）
- final judgment（最终判断）

AI 是高速工程执行器，不是 autonomous system architect（自主系统架构师）。

---

## 3. Stable Core + High-Velocity Research Edge

本项目必须同时保护两类区域：

```text
Stable Core
+
High-Velocity Research Edge
```

### 3.1 Stable Core

Stable Core 是慢变化、高审查、contract-first 的系统核心，包括：

- `EvaluationSpec`
- artifact schema
- `schema_registry`
- `manifest.display`
- evidence flow
- time semantics
- market semantics
- API-visible identity，例如 `(source_alias, run_id)`
- public API payloads
- frontend shared contracts
- directory boundaries

这些区域不应该被 AI 随手修改。任何涉及 contract、schema、public surface 或语义变化的修改，都必须有明确范围、blast radius check 和人工批准。

### 3.2 High-Velocity Research Edge

High-Velocity Research Edge 是允许快速试错和 AI 高速执行的区域，包括：

- `research/factors`
- Factor Factory candidate generation
- experiments
- evaluation metrics 的局部扩展
- feature engineering
- notebooks / scratch / explore
- local reports / smoke tests
- feature-local frontend UI

这些区域可以快速迭代，但不能污染 Stable Core。

---

## 4. Contracts Are the Real System

本项目真正稳定的不是某个具体实现，而是：

```text
contracts
schemas
semantics
evidence flow
```

实现可以被替换、优化或重构；contracts 不能静默漂移。

核心原则：

- implementations are replaceable
- contracts are governed
- schema drift 必须被显式发现
- semantic drift 比字段名变化更危险
- display contract 必须统一
- historical artifacts 不应靠放宽 loader 规则继续可见；应重新生成或明确标注历史状态

任何涉及以下内容的变更都属于高风险：

- artifact schema
- `manifest.display`
- API payload
- frontend shared types
- evaluation semantics
- time semantics
- run identity
- schema registry
- generated artifact layout

---

## 5. Evidence-Flow-First Architecture

当前核心证据链是：

```text
raw / interim / panel / factors
-> evaluations
-> artifacts
-> manifest.display
-> API / downloads
-> frontend evidence workbench
```

规则：

- frontend 不重新计算 IC、RankIC、returns、weights、turnover、eligibility 或 tradability
- API 不绕过 `manifest.display` 暴露用户可见文件
- artifact writer 不输出 frontend-specific shortcut
- frontend 不直接读取本地文件
- download / preview / chart / table 都必须通过 display contract

`manifest.display` 是用户可见展示合同。任何 artifact/API/frontend 相关变更必须同步：

- producer
- writer
- loader
- schema registry
- manifest/display
- API payload / route / service
- downloads / ZIP
- frontend consumer
- tests
- docs
- code-map

---

## 6. Future Capabilities Shape Boundaries, Not Current Complexity

未来能力可以影响边界设计，但不能提前拉高当前实现复杂度。

正确做法：

```text
因为未来要支持多市场/多频率/高频，所以现在拆清 datasets / markets / research / evaluations / artifacts 边界。
```

错误做法：

```text
因为未来要支持高频，所以现在提前实现 tick replay engine、distributed scheduler、live execution、plugin framework。
```

长期能力可以写入 roadmap / target architecture，但不能自动进入当前代码。以下内容在当前阶段默认不实现：

- distributed runtime
- workers / schedulers
- billing / compliance
- live trading
- feature store
- model registry
- autonomous agents
- broker / exchange connector
- event-driven simulation
- generalized orchestration framework

---

## 7. Evaluations != Simulations

必须严格区分：

```text
evaluations
```

和：

```text
simulations
```

`evaluations/` 负责：

- factor evaluation
- prediction evaluation
- weight / portfolio diagnostics
- IC / RankIC
- grouped returns
- turnover
- portfolio return diagnostics
- cost-aware descriptive diagnostics
- schema 0.7 evaluation-set artifacts

`simulations/` 未来才负责：

- order lifecycle
- fills
- latency
- slippage
- market replay
- event-driven backtest
- paper exchange
- execution model

`EvaluationSpec` 不是 `BacktestSpec`，也不是 `SimulationSpec`。不要把高频、撮合、成交、订单生命周期、真实交易语义硬塞进当前 evaluation contract。

---

## 8. N5 的真实状态：Partial Implementation + Contract Groundwork

N5 不再是纯 planned capability。当前更准确的状态是：

```text
N5 partial implementation + contract groundwork
```

已经落地的是：

- schema 0.7 diagnostics framework
- diagnostics registry
- required diagnostics tables
- conditional materialized views
- `diagnostic_summary`
- `diagnostic_metrics`
- `manifest.display` integration
- API / frontend evidence flow
- descriptive proxy diagnostics
- year / month IC breakdown artifacts

但完整 N5 仍是 future work，包括：

- true risk decomposition
- real capacity modeling
- PIT regime engine
- neutralized diagnostics
- execution-aware simulation
- slippage / fill / replay semantics
- independent `BacktestSpec` / `SimulationSpec`

因此，文档中应区分：

```text
N5 v1 descriptive proxy diagnostics
```

和：

```text
full N5 risk/regime/cost/capacity capability
```

不要把 descriptive diagnostics 写成正式 risk model、真实 capacity analysis、simulation 或 trading capability。

---

## 9. Refactor Governance

重构必须由真实问题触发，而不是由“更优雅”“更通用”“未来可能用得上”触发。

允许重构的典型触发条件：

- 同一模式第三次出现
- 当前结构导致 bug 重复出现
- 测量确认的性能瓶颈
- 当前结构阻碍局部扩展
- contract 已稳定
- tests 已经覆盖行为

禁止：

- feature + refactor 混合提交
- schema change + refactor 混合提交
- directory restructuring 作为顺手操作
- 抽出未被真实需求验证的 generic framework
- 为 future capability 提前创建空实现层

默认原则：

```text
prefer explicit implementation over premature abstraction
```

---

## 10. Abstraction Creep 与 Framework Creep

AI 很容易制造看起来合理但实际增加长期维护成本的抽象，例如：

- `Base*`
- `Abstract*`
- `Manager`
- `Registry`
- `Factory`
- `Plugin`
- `Framework`
- generalized orchestration

这些不是绝对禁止，但必须满足：

- 重复模式已经出现多次
- contract 已稳定
- 多个模块有真实共同需求
- tests 能保护行为
- 当前实现已经明显阻碍开发
- 人类明确批准

否则，保持显式实现。

---

## 11. Task Mode Principles

本项目的 AI / Codex 工作应区分任务类型，但本文只保留原则。具体 task-mode templates、recommended reasoning effort、context budget 和执行步骤由 `AGENTS.md`、maintenance skill、context-routing、`.codex/templates/` 和 current brief 维护。

### 11.1 Goal Clarity

所有 medium+ 工作都应先明确 goal、current phase、allowed scope、forbidden scope、success condition、non-goals、relevant contracts 和 expected blast radius。

很多 drift 来自 goal 不清晰，而不是实现能力不足。

### 11.2 Decision Before Implementation

新 contract、新 schema、新架构、重大 refactor 和 future capability 必须先完成 decision / planning，再进入 implementation。AI 不应把需要人类判断的设计问题伪装成普通改代码任务。

### 11.3 Governed Execution

边界清楚、无需逐步决策的一组任务可以用 governed execution 推进，但必须受明确 envelope、stop conditions 和 verification plan 约束。遇到 contract ambiguity、schema semantics change、new abstraction / framework、architecture expansion、future capability implementation、unclear blast radius 或 execution / simulation semantics 时必须停止。

### 11.4 Audit / Assessment Boundaries

文档、治理、路线图、合同和架构审计应先区分 discovery-only、assessment、docs-sync 和 implementation。审计结论不应自动升级为跨层实现，除非任务 envelope 已批准该范围。

---

## 12. Approval Gate

需要人类判断的 medium+ tasks 必须在实现前完成 task brief、implementation plan、scope declaration 和 explicit human approval。具体 approval gate 的格式和何时等待，以 `AGENTS.md`、maintenance skill 和 `.codex/templates/` 为准。

但 approval gate 不应卡住所有小任务。小范围、低风险、无 contract 变化、无跨层影响的任务，可以按 governed batch 执行。

目标不是让 AI 每一步都问人，而是让 AI 在真正涉及决策时停下来。

---

## 13. Context Loading

大型项目中，更多 context 不一定更好。过量 context 会诱导：

- architecture drift
- abstraction creep
- future capability contamination
- scope expansion

推荐原则：

- start narrow from the task surface
- use current brief for short current orientation
- load task contracts before broad architecture
- load implementation files and direct dependencies before speculative future docs
- load future roadmap only for architecture / planning / future capability tasks

默认不要让 AI “看完整项目后自由发挥”。

---

## 14. Documentation System Principles

文档系统必须分层，否则会出现职责重叠和 narrative drift。

### 14.1 Active Operational Docs

Active docs 只记录 current operational truth。

包括：

- `README.md`
- `PROJECT_STATUS.md`
- `EVALUATION_INTERFACE.md`
- `docs/code-map/*`

它们必须跟当前实现同步。

### 14.2 Governance Docs

Governance docs 记录规则，不记录一次性进度。

包括：

- `AGENTS.md`
- `.codex/skills/quant-research-platform-maintenance/SKILL.md`
- `engineering-standards.md`
- `quant-research-standards.md`

### 14.3 Architecture Docs

Architecture docs 记录边界和长期结构。

包括：

- `ARCHITECTURE.md`
- `target-architecture-and-capabilities.md`
- `docs/architecture/README.md`

它们不应承担当前任务流水。

### 14.4 Roadmap Docs

Roadmap docs 只记录未完成或 active roadmap。

Completed closeouts 不应长期留在 active roadmap 中。

### 14.5 Research Docs

Research docs 回答研究问题，不承担工程状态同步。

- `REPORT.md` = research narrative
- `RESEARCH_CARD.md` = evidence index

### 14.6 Historical Docs

Historical docs 不应被改写成最新状态，而应有醒目的 historical boundary marker。

例如：

```text
本文记录某一时间点的历史冻结状态。
当前运行合同以 README、EVALUATION_INTERFACE、schema registry、code-map 和 tests 为准。
```

---

## 15. Code-Map as Current Implementation Learning Layer

`docs/code-map/` 的正式定位是：

```text
Current Implementation Learning Layer
```

它负责：

- current implementation reality
- runtime flow
- entrypoints
- module relations
- artifact/API/frontend tracing
- onboarding / developer learning
- “当前系统怎么工作”

它不负责：

- governance
- roadmap
- future architecture
- historical closeout
- project status
- research narrative
- completed milestone history

当前不需要新增 `docs/developer/`。如果未来要建立 developer docs，必须证明 code-map 已经无法承载当前实现学习需求。

---

## 16. Active vs Historical Separation

Active docs 只保留 current operational truth。

Historical docs 保留历史事实，但必须冻结边界。

不要把历史文档全部改成最新 schema；那会破坏设计考古价值。正确做法是：

```text
保留历史事实
+
增加历史边界说明
```

例如 schema 0.5 的 Phase 1 acceptance record 不应改成 schema 0.7，但必须明确 schema 0.5 是当时验收事实，当前运行合同已演进。

---

## 17. Maintenance Skill 的角色

`.codex/skills/quant-research-platform-maintenance/SKILL.md` 应沉淀已经验证过的稳定 workflow，例如：

- medium+ approval gate
- task brief
- review summary
- scope declaration
- artifact/API/frontend sync rule
- active docs rule
- code-map role

不要把所有治理理论、复杂 escalation matrix、future workflow 都塞进 skill。

Skill 是操作层，不是哲学层。

---

## 18. Velocity vs Governance

治理的目标不是让开发变慢，而是减少返工。

正确状态是：

```text
enough governance to prevent drift
+
enough freedom to preserve research velocity
```

过少治理会导致：

- scope creep
- hidden contract drift
- half migration
- future contamination

过度治理会导致：

- prompt 过长
- Codex 困惑
- velocity 崩塌
- governance itself becomes drift source

因此，本项目采用：

```text
progressive governance
```

即只把重复出现、影响长期演化的高频问题固化为规则。

---

## 19. 什么情况必须停止并请求人类确认

AI 必须停止的情况：

- contract meaning unclear
- schema semantics unclear
- task touches multiple architecture layers
- new abstraction/framework/dependency is proposed
- work crosses current phase boundary
- public contract changes
- directory structure changes
- future capability implementation is implied
- evaluation begins to become simulation
- blast radius is unclear

---

## 20. 最终原则

本项目的 AI-native engineering baseline 可以概括为：

```text
Stable Core
High-Velocity Research Edge
Contract-first
Evidence-flow-first
Current Phase Awareness
Human Approval for Decisions
Governed Batch for Execution
Code-map as Implementation Learning Layer
Active Docs for Current Truth
Historical Docs with Explicit Boundaries
Evaluations != Simulations
Future Capabilities Shape Boundaries, Not Current Complexity
```

这个文档用于恢复长期 mental model。  
具体执行仍以项目内当前 rules、contracts、tests 和 code-map 为准。
