# AI-Native Engineering Standard v1.0

> 本文是 `quant-research-platform` 的 AI-native engineering full governance reference / complete standard。
> 它定义人类、Codex、GPT Pro 和其他 AI agent 如何协作维护一个长期演化的本地量化研究平台。
>
> 本文不是日常任务 prompt，也不是 Codex 每次执行都必须完整读取的上下文。
> 日常执行边界以项目内 `AGENTS.md`、`.codex/skills/quant-research-platform-maintenance/SKILL.md`、`.codex/templates/*`、`.codex/briefs/current-qrp-status.md`、代码、tests、`EVALUATION_INTERFACE.md` 和 `docs/code-map/` 为准。
>
> 本文保留完整原则、治理模型、task-mode 体系和长期演化规则；短任务应使用 templates，普通任务不应把本文全文塞进 prompt。

---

## 0. Purpose and Authority / 目的与权限

### 0.1 Purpose

本标准的目标不是让 AI 开发变慢，而是让 AI-native engineering 可持续：

```text
high engineering velocity
+
contract safety
+
private-data safety
+
architectural discipline
+
research iteration speed
```

它解决的问题包括：

- Codex 大范围读取 repo 导致 token / quota 消耗失控。
- AI 在上下文过载时发生 scope creep、abstraction creep、future capability contamination。
- contract / schema / API / frontend / artifact 同步不完整。
- 文档、roadmap、code-map、skills、templates 职责混乱。
- 长任务 `/goal` 与小任务使用同一套 prompt，导致低风险任务过度治理，高风险任务又缺乏明确 envelope。

### 0.2 Authority Model

本文是 full governance reference，不是单次任务执行层。

执行优先级：

```text
1. Human direct instruction / explicit task envelope
2. AGENTS.md hard boundaries
3. Project maintenance skill workflow
4. Task templates and current brief
5. Code, tests, schema registries, manifest.display, EVALUATION_INTERFACE, code-map
6. This full standard as background governance reference
7. Future roadmap and target architecture
```

规则放置原则：

| Layer | Owns |
|---|---|
| `AGENTS.md` | hard boundaries, private-data safety, phase boundary, forbidden actions |
| project skill | operational workflow, review gates, task modes, contract maintenance playbooks |
| `.codex/templates/*` | task execution prompts and context budgets |
| `.codex/briefs/current-qrp-status.md` | short current status for ordinary Codex tasks |
| code/tests/schema registries | enforceable behavior and contracts |
| docs/code-map | current implementation learning layer |
| this standard | full philosophy and governance reference |

### 0.3 Non-Goals

本文不应成为：

- 每次 Codex prompt 的大段复制内容。
- 当前 task status 的存放处。
- 一次性 run result、debug log、commit hash 清单或临时计划。
- 替代 tests / schema registry / `manifest.display` 的软约束。

---

# Part 1 — Core Philosophy

## 1. System Identity

`quant-research-platform` 不是：

```text
单次量化研究脚本
单次回测项目
单市场策略仓库
SaaS 产品
paper trading 系统
live trading 系统
生产交易平台
```

它的真实定位是：

```text
AI-native Quant Research Platform
```

当前阶段：

```text
Current Phase — Local Research Platform
```

当前核心目标：

- factor research（因子研究）
- evaluations（评价）
- schema artifacts（产物合同）
- `manifest.display`（展示合同）
- evidence workflow（证据链）
- local evidence workbench（本地证据工作台）
- A-share research reliability（A 股研究可靠性）
- Factor Factory / A-share pipeline 的本地研究效率

当前阶段必须避免把 future capability 写成当前能力。

## 2. AI Accelerated, Human Governed

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
- bounded batch execution（受控批量执行）
- approved goal / overnight execution（批准包络内的长任务执行）

人类负责：

- contracts（合同）
- architecture boundaries（架构边界）
- phase control（阶段控制）
- abstraction level（抽象层级）
- priority（优先级）
- scope approval（范围批准）
- risk acceptance（风险接受）
- final judgment（最终判断）

AI 是高速工程执行器，不是 autonomous system architect。

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
- `PipelineConfig` / public pipeline semantics
- artifact schema
- `schema_registry`
- `manifest.display`
- evidence flow
- time semantics
- market semantics
- dataset contracts
- API-visible identity，例如 `(source_alias, run_id)`
- public API payloads
- frontend shared contracts
- directory boundaries
- evaluation vs simulation boundary
- private-data exposure boundary

这些区域不应被 AI 随手修改。任何涉及 contract、schema、public surface 或语义变化的修改，都必须有明确 scope、blast radius check、tests 和人工批准。

### 3.2 High-Velocity Research Edge

High-Velocity Research Edge 是允许快速试错和 AI 高速执行的区域，包括：

- `research/factors`
- Factor Factory candidate generation
- experiments
- feature engineering
- evaluation metrics 的局部扩展
- notebooks / scratch / explore
- feature-local frontend UI
- docs/code-map 的实现事实同步
- fixture-based tests

这些区域可以快速迭代，但不能污染 Stable Core。

## 4. Contracts Are the Real System

本项目真正稳定的不是某个具体实现，而是：

```text
contracts
schemas
semantics
evidence flow
```

核心原则：

- implementations are replaceable
- contracts are governed
- schema drift 必须被显式发现
- semantic drift 比字段名变化更危险
- display contract 必须统一
- tests / schema registries 是可执行真相源
- historical artifacts 不应靠放宽 loader 规则继续可见；应重新生成或明确标注历史状态

高风险变更包括：

- artifact schema
- `manifest.display`
- API payload
- frontend shared types
- evaluation semantics
- time semantics
- run identity
- schema registry
- generated artifact layout
- private-data exposure surface

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
- generated artifacts 是 evidence，不是任意任务上下文

任何 artifact / API / frontend 相关变更必须同步：

- producer
- writer
- loader
- schema registry
- `manifest.display`
- API payload / route / service
- downloads / ZIP
- frontend consumer
- tests
- docs
- code-map

## 6. Future Capabilities Shape Boundaries, Not Current Complexity

未来能力可以影响边界设计，但不能提前拉高当前实现复杂度。

正确做法：

```text
因为未来要支持多市场 / 多频率 / 高频，所以现在拆清 datasets / markets / research / evaluations / artifacts 边界。
```

错误做法：

```text
因为未来要支持高频，所以现在提前实现 tick replay engine、distributed scheduler、live execution、plugin framework。
```

当前阶段默认不实现：

- distributed runtime
- workers / schedulers
- billing / compliance
- live trading
- paper trading
- feature store
- model registry
- broker / exchange connector
- external vendor automation
- event-driven simulation
- generalized orchestration framework

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
- evaluation-set artifacts

`simulations/` 如果未来设计，才负责：

- order lifecycle
- fills
- latency
- slippage
- market replay
- event-driven backtest
- paper exchange
- execution model

`EvaluationSpec` 不是 `BacktestSpec`，也不是 `SimulationSpec`。不要把撮合、成交、订单生命周期、真实交易语义硬塞进当前 evaluation contract。

---

# Part 2 — Usage Efficiency and Task Modes

## 1. Usage Efficiency Principle

本项目的 Codex usage efficiency 目标不是让所有任务最低 token，而是：

```text
在默认高质量模型的前提下，最大化单位会员额度 / 单位工程时间内的有效工程闭环。
```

优化重点：

- 低风险任务低 token 化
- 普通任务窄上下文化
- 高价值 `/goal` / overnight 任务受控高 token 化
- 避免重复读取 AGENTS / skill / docs / repo / tests / diff / logs
- 避免普通任务误触发大上下文审计
- 把高成本请求留给真正需要判断力的任务

不应为了省 token 删除：

- private-data safety
- stop conditions
- challenge / objection mechanism
- commit message recommendation
- subagent governance
- Scope Check
- targeted tests
- documentation sync gate

## 2. Default Model Assumption

默认模型可以保持高质量主力模型。用户不应被要求频繁切模型。

主要控制手段是：

```text
task mode
recommended reasoning effort
context budget
file-read budget
subagent policy
staged workflow
```

模型切换可以作为可选优化，但不应成为日常工作流的必要前提。

## 3. Three-Layer Task System

### 3.1 Fast Path

用于低风险、机械或 docs-only 任务：

- discovery-only
- docs-sync
- commit-and-push
- branch-push
- ff-merge-and-push
- git status / diff name-only
- small wording cleanup

默认：

```text
Recommended reasoning effort: Low
No subagent
No broad repo loading
No contracts reference unless needed
No roadmap / target architecture / archive
```

### 3.2 Standard Governed Task

用于普通开发闭环：

- focused-audit
- focused-implementation
- pre-commit-review
- small MVS
- fixture-based tests
- pipeline/API/artifact-adjacent local work

默认：

```text
Recommended reasoning effort: Medium / Medium-High
Explicit allowed files
Explicit forbidden surfaces
File budget
Targeted tests
Diff review before commit
```

### 3.3 Goal / Overnight

用于高价值长任务：

- `/goal`
- overnight run
- multi-phase implementation
- review-and-fix-loop
- cross-surface but approved execution
- major contract decision implementation after approval

默认：

```text
Recommended reasoning effort: High / Highest available
High token allowed
Broader context allowed inside approved envelope
Subagents allowed with narrow context packets
Review-and-fix loop allowed
Verification cadence required
Checkpoint report required
```

Goal / Overnight 不是省 token 模式。它是**受控高 token 模式**。

## 4. Reasoning Effort Recommendation

模板必须自动给出 reasoning effort，减少用户判断负担。

| Task mode | Recommended reasoning effort |
|---|---|
| push-only / branch-push / commit-and-push / ff-merge-and-push | Low |
| docs-sync | Low |
| discovery-only | Low |
| focused-audit | Medium; High if contract/private/API/schema boundary |
| focused-implementation | Medium; Medium-High if pipeline/API/artifact-adjacent |
| pre-commit-review | Medium |
| decision-audit | High |
| goal-overnight | High / Highest available |
| schema migration / major architecture decision | Highest available |

## 5. Context Budget Policy

### 5.1 Fast Path Budget

```text
0-3 files
rg / git metadata first
no full repo
no broad docs
```

### 5.2 Standard Task Budget

```text
4-8 files by default
up to 12 only after reporting why
read line ranges before full files
use rg first
load direct tests only
```

### 5.3 Pre-commit Review Budget

```text
diff-only by default
changed files only when necessary
changed tests only
one review-only subagent if justified
```

### 5.4 Goal / Overnight Budget

Goal / Overnight 不使用普通小预算，但必须有 context loading plan：

- which rules to load
- which code maps to load
- which contracts to load
- which directories are in scope
- which directories are explicitly out of scope
- which files/subsystems each subagent may inspect

Goal mode 可以扩大上下文，但不能 blind-load entire repo without phase plan。

## 6. Prompt Caching Reality

缓存命中不是免费记忆。

正确理解：

```text
repeated prefix can be cheaper
but repeated context still counts
```

因此优化目标不是盲目提高 cache hit，而是减少不必要的 repeated context。

---

# Part 3 — AI Governance and Permissions

## 1. Governance Model

本项目采用：

```text
Governed AI Coding
```

不采用：

```text
Autonomous AI Coding
```

AI 可以高速执行，但不能自主扩大系统范围、改写核心合同或决定阶段升级。

## 2. Layered Permission Model

### 2.1 High Permission Zone

允许 AI 高速生成和局部重构：

```text
research/factors
factor_factory candidate generation
experiments
feature engineering
tests
frontend feature-local UI
pipeline configs
evaluation metrics local extensions
docs/code-map implementation facts
```

### 2.2 Medium Permission Zone

允许 AI 局部修改，但需要 scope review：

```text
datasets readers
markets/implementations/a_share
pipeline internals
runtime helpers
API service internals
frontend feature state
docs/code-map current implementation tracing
```

禁止 cross-layer redesign。

### 2.3 Low Permission Zone

AI 默认不得自动修改：

```text
EvaluationSpec
PipelineConfig public semantics
schema_registry
artifact_schema
manifest.display
public API payloads
frontend shared types
directory structure
time semantics
market semantics
evaluation vs simulation semantics
```

修改必须有 human approval、contract blast radius plan 和 tests。

### 2.4 Restricted Zone

必须人工批准：

```text
new market
new frequency model
simulation semantics
repo-level apps/
distributed runtime
live trading
paper trading
new dependency
external API/vendor integration
contract-breaking changes
directory restructuring
```

## 3. AI Stop Conditions

以下情况 AI 必须停止并请求人类决策：

- contract ambiguity
- schema semantics unclear
- cross-layer impact
- new abstraction / framework
- new dependency
- phase boundary crossing
- public contract change
- directory restructuring
- large or unclear blast radius
- evaluation begins to become simulation
- future capability implementation is implied
- private data / path exposure risk
- destructive filesystem behavior
- unexpected generated/private artifacts

## 4. Challenge / Objection Mechanism

如果 AI 认为用户 prompt 约束错误、过窄、与代码/测试/合同冲突，或会降低项目质量，必须先提出反驳，而不是静默执行或静默违反。

格式：

```text
Proposed Challenge / Objection
- challenged constraint:
- evidence from code/docs/tests:
- why constraint may be wrong:
- affected surfaces:
- recommended alternative:
- whether human approval is required:
```

未经用户批准，不得按替代方案实现。

## 5. Task Size Budget

Small task：

```text
1-5 files
single layer
single intent
```

Medium task：

```text
5-15 files
single contract domain
```

Large task：

```text
15+ files
cross-layer
schema-affecting
future capability boundary
```

Large task 必须有 task brief、phase plan、verification plan、stop conditions、human approval。必要时需要 ADR。

## 6. Contract Blast Radius Check

修改 schema / artifact / manifest / API payload / shared type 时，必须列出：

- affected layers
- affected tests
- affected frontend contracts
- affected API routes
- affected loaders/writers
- affected downloads
- affected inventory/index flows
- affected docs/code-map
- private-data exposure risk

## 7. AI Drift Definition

AI drift 包括：

- Architecture Drift
- Abstraction Drift
- Scope Drift
- Contract Drift
- Future Capability Drift
- Narrative Drift
- Documentation Drift
- Context Drift
- Verification Drift

---

# Part 4 — Phase Governance

## 1. Current Phase

当前项目阶段：

```text
Current Phase — Local Research Platform
```

当前 in scope：

- research / factors
- Factor Factory
- candidate generation
- local A-share data pipeline
- evaluation artifacts
- evidence workbench
- local API as research helper
- documentation and code-map synchronization
- reliability of local research workflow

## 2. Current Phase Out of Scope

当前阶段 out of scope：

- live trading
- paper trading
- broker / exchange API
- external vendor automation
- web data API integration
- deployed service
- distributed scheduler / worker
- model registry
- feature store
- event-driven simulation
- generalized plugin / orchestration framework

## 3. Future Capability Rule

Future capability 可以影响边界设计，但不能自动进入当前代码。

## 4. Multi-Market Governance

A-share 可以是第一个完整 market module；其他 markets 只能在明确设计、数据边界、tests 和 approval 后进入。禁止空实现包作为 planning marker。

## 5. Multi-Frequency Governance

多频率设计可以影响 abstraction boundary，但不能让当前日频/中低频研究系统提前实现 tick replay、order lifecycle 或 execution engine。

## 6. Simulation Boundary Rule

Simulation 是 future domain。Evaluation 不得被逐步污染成 hidden simulation。

## 7. Speculative Infrastructure Ban

禁止为“未来可能需要”提前实现框架、插件系统、调度系统、runtime agents 或 deployable app shell。

## 8. Phase Upgrade Rule

任何 phase upgrade 必须有明确目标、风险边界、acceptance criteria、rollback plan 和 human approval。

## 9. Local-First Principle

当前系统优先保持 local, inspectable, reproducible。不要为远期 deployment 牺牲当前研究效率。

## 10. Research Velocity Protection

治理目标是减少返工，不是让研究停滞。对 High-Velocity Research Edge，应允许快速局部迭代，只要不污染 Stable Core。

---

# Part 5 — Contract Governance

## 1. Contract-Centric Architecture

核心合同包括：

- evaluation input contract
- artifact schema
- display contract
- API-visible identity
- frontend shared types
- factor output format
- pipeline start semantics
- private-data boundary

## 2. Contract Freeze Levels

| Level | Description | Change rule |
|---|---|---|
| L0 exploratory | scratch / notebooks / local experiments | free to change |
| L1 internal | private helper / local implementation | tests recommended |
| L2 module contract | function/class used across module boundary | tests required |
| L3 public project contract | schema/API/frontend/artifact/display | explicit approval + blast radius |
| L4 phase boundary | market scope / simulation / deployment / trading | architecture decision required |

## 3. `manifest.display` Governance

`manifest.display` 是用户可见展示合同。

规则：

- API/download/frontend 不绕过它。
- 新增用户可见 table/chart/download 必须同步 writer/loader/display/API/frontend/tests/docs。
- 不为前端临时推导 artifact path。
- display contract 中的 path 必须 portable、relative、safe。

## 4. Evidence Flow Governance

证据链必须保持：

```text
calculation -> artifact -> manifest.display -> API -> frontend
```

不得：

- frontend 重新计算 evidence
- API 使用 hidden shortcut
- artifacts 写 frontend-specific fields
- docs 声称未实现能力

## 5. Semantic Compatibility Rule

字段名兼容不等于语义兼容。任何 time semantics、label definition、return source、eligibility/tradability、cost、portfolio construction 语义变化都必须显式记录并测试。

## 6. Time Semantics Governance

必须明确：

- decision time
- as-of time
- label horizon
- execution lag
- rebalance cadence
- leakage controls

## 7. Market Semantics Governance

Market-specific rules 不能隐藏在 generic layer。A-share 的停牌、涨跌停、ST、上市天数、T+1、印花税、流动性约束等必须在市场层或明确的 evaluation context 中表达。

## 8. Schema Registry Governance

Schema registry 和 tests 是字段、grain、unique key、conditional rules、required files 的执行真相源。Markdown 只能解释，不应成为唯一 enforcement。

## 9. Half Migration Ban

禁止 producer、loader、API、frontend、tests、docs 只迁一半。合同变更必须 linked update。

## 10. Breaking Change Protocol

Breaking change 必须包含：

- reason
- affected surfaces
- migration or regeneration plan
- retired behavior coverage
- tests
- docs sync
- rollback risk

## 11. Artifact Identity Governance

API-visible identity 必须稳定。不要把 local path、source-root、runtime host、cache path 写入 portable artifact metadata。

## 12. Local Filesystem Governance

所有 reusable docs/code/config 应使用 project-relative path。禁止写入 drive letters、home directories、user names、absolute private paths。

## 13. No Test, No Main

Code / contract / behavior-changing work 必须有相关 tests 或 named static checks。

Docs-only change 可使用：

- `git diff --check`
- changed-file review
- link / scope review
- targeted `rg`

不要求 docs-only change 跑完整 Python/Vitest suite。

---

# Part 6 — Task Protocol

## 1. Task-Oriented Governance

AI 工作必须 task-oriented，而不是 continuous autonomous evolution。

每个任务必须明确：

```text
goal
scope
contracts
forbidden scope
checks
definition of done
```

## 2. Required Task Brief

Medium+ task 必须先生成 task brief 或使用对应 template。

Full brief 包含：

```text
Goal
Task mode
Current phase
Allowed scope
Forbidden scope
Relevant contracts
Files to inspect
Expected changes
Checks required
Definition of done
Risks
Stop conditions
```

Small / Fast Path task 可以使用精简模板。

## 3. Task Modes

### 3.1 Discovery-Only

用于定位文件和风险 surface。

规则：

```text
Low effort
rg first
no edits
no broad loading
candidate files only
```

### 3.2 Docs Sync

用于少数 docs 状态同步。

规则：

```text
Low effort
target docs only
no src/tests
no full contract reference unless required
```

### 3.3 Focused Audit

用于单 surface 审查。

规则：

```text
Medium / High effort
explicit file budget
no implementation
output minimal fix / MVS
```

### 3.4 Focused Implementation

用于已批准范围内的局部实现。

规则：

```text
Medium / Medium-High effort
explicit allowlist
targeted tests
no broad refactor
```

### 3.5 Pre-commit Review

用于提交前审查。

规则：

```text
Medium effort
diff-only by default
review-only subagent allowed if justified
fix confirmed in-scope blocker only
```

### 3.6 Commit / Push / Merge Tasks

`commit-and-push`：确认当前 branch 改动后 commit 并 push。

`branch-push`：已有 commit，只 push feature branch。

`ff-merge-and-push`：fast-forward merge 到 main 并 push main。

默认禁止：

- force push
- amend
- rebase
- merge commit
- squash

除非用户明确批准。

### 3.7 Goal / Overnight

用于受控长任务。

必须有 approved envelope：

```text
Goal
Allowed scope
Forbidden scope
Context loading plan
Phase plan
Subagent plan
Verification cadence
Checkpoint report cadence
Commit / push / merge policy
Stop conditions
Final report format
```

Goal mode 可以高 token，但不能绕过安全边界。

## 4. Implementation Plan Requirement

Medium / large / cross-layer / contract-touching task 必须先写 implementation plan：

```text
Goal
Affected files
Unaffected contracts
Execution steps
Test plan
Rollback risk
Completion criteria
```

## 5. Scope Declaration

AI 必须明确 allowed scope 和 forbidden scope。单次 prompt 中必须保留任务差异：本轮目标、允许文件、本轮额外禁止、checks、commit/push policy。

## 6. File Inspection Protocol

AI 不允许 blind coding。Coding 前必须读取 relevant contracts、tests、code-map 和 direct implementation files。

但读取必须按 context budget 控制：

- ordinary task 不 broad-load repo
- use `rg` first
- prefer line ranges
- exceeded file budget requires report and approval

## 7. Review Summary Requirement

任务结束后输出应至少包含：

```text
Changed / Findings
Not Changed
Checks Run
Scope Check
Residual Risks / Known Limitations
Recommended commit message or Next Step
```

具体格式由 task template 决定。

## 8. Scope Check Requirement

Scope Check 必须覆盖相关项：

```text
no schema change
no manifest.display change
no API/frontend behavior change
no private artifacts
no unapproved files
no lowered safety boundary
```

Git tasks 还必须说明：

```text
pushed branch/main
no force push
no rebase
no amend
no merge commit unless approved
```

## 9. Human Approval Required

以下情况必须人工批准：

- new architecture
- new contract
- new dependency
- new phase capability
- schema redesign
- directory restructuring
- simulation semantics
- distributed runtime
- deployment / trading expansion
- private generated artifact handling

---

# Part 7 — Goal / Overnight and Review-and-Fix Loop

## 1. Goal Mode Principle

Goal mode 是高价值长任务模式，不是省 token 模式。

Goal mode 允许：

- larger context
- multiple phases
- multiple subagents
- implementation + review + fix loop
- broader verification
- checkpoint reporting

前提是 approved envelope 明确。

## 2. Approved Envelope

Goal / Overnight 开始前必须写明：

```text
Goal:
Why this matters:
Branch/worktree plan:
Allowed scope:
Forbidden scope:
Context loading plan:
Phase plan:
Subagent plan:
Verification cadence:
Commit/push policy:
Stop conditions:
Final report format:
```

## 3. Phase Plan

Goal mode 必须分 phase：

```text
Phase 1: discovery / audit
Phase 2: implementation slice
Phase 3: focused tests
Phase 4: review-and-fix
Phase 5: docs sync
Phase 6: commit / push if approved
```

Phase 可根据任务调整，但不能无限滚动。

## 4. Review-and-Fix Loop

Review-and-fix loop 规则：

- Review should focus on confirmed P0/P1/P2 issues.
- P0/P1/P2 inside approved scope may be fixed without asking again.
- P3 cleanup should be recorded or fixed only if small and in scope.
- After each fix, rerun the narrowest relevant check.
- If new issue classes keep appearing, stop and request human decision.

## 5. Checkpoint Report

Long-run tasks 应阶段性输出：

```text
Progress
Files Changed
Checks Run
Risks / Blockers
Next Phase
```

Checkpoint 不应泄漏 private paths、row-level data 或 generated private artifacts。

## 6. Goal Stop Conditions

Goal mode 必须在以下情况停止：

- approved envelope 不足
- schema / manifest.display / public API / frontend contract 需要未批准变更
- private data / path exposure risk
- new dependency / abstraction / directory restructuring
- task drifts into deployment / trading / simulation
- tests fail with unclear cause
- generated/private artifacts unexpectedly appear
- blast radius becomes unclear

---

# Part 8 — Refactor Governance

## 1. Refactor Definition

Refactor 是 behavior-preserving structural improvement，不改变 contract、semantics 或 evidence flow。

## 2. Refactor Is Triggered, Not Emotional

允许重构的原因：

- 同一模式第三次出现
- 性能问题已测量证实
- 当前结构无法局部扩展
- 相同错误反复出现
- contract 已稳定
- tests 已覆盖行为

禁止由“更优雅”“更先进”“更通用”驱动。

## 3. Mega Refactor Ban

禁止一次性“重构整个 evaluation system”“统一所有 market abstraction”“升级 artifact architecture”“重做 pipeline framework”。

## 4. One Refactor, One Intent

Refactor 必须 single-purpose。禁止 refactor + feature + schema redesign + directory restructuring 混合。

## 5. Refactor Isolation Rule

Refactor 应独立 branch、独立 review、独立 test verification。

## 6. Abstraction Creep Governance

AI 天然倾向：

```text
Base
Abstract
Manager
Registry
Factory
Plugin
Framework
Orchestrator
```

默认禁止 premature abstraction。

抽象必须满足：

- 重复模式已经出现多次
- contract 已稳定
- 多个模块有真实共同需求
- tests 能保护行为
- 当前实现已明显阻碍开发
- 人类明确批准

## 7. Framework Creep Ban

禁止 domain code 演化成 generalized infrastructure engineering，除非有明确阶段升级和架构批准。

## 8. Directory Refactor Governance

目录结构属于 architectural surface。禁止 AI spontaneous directory restructuring。

---

# Part 9 — Documentation Topology

## 1. Active Operational Docs

Active docs 只记录 current operational truth。

包括：

- `README.md`
- `PROJECT_STATUS.md`
- `EVALUATION_INTERFACE.md`
- `docs/code-map/*`
- current brief

它们必须跟当前实现同步。

## 2. Governance Docs

Governance docs 记录规则，不记录一次性进度。

包括：

- `AGENTS.md`
- project maintenance skill
- engineering / quant standards
- this standard

## 3. Architecture Docs

Architecture docs 记录边界和长期结构。

包括：

- `ARCHITECTURE.md`
- `target-architecture-and-capabilities.md`
- architecture README / principles

它们不应承担当前任务流水。

## 4. Roadmap Docs

Roadmap docs 只记录 unfinished work 和 active priorities。Completed closeout 不应长期留在 active roadmap 中。

## 5. Research Docs

Research docs 回答研究问题，不承担工程状态同步。

- `REPORT.md` = research narrative
- `RESEARCH_CARD.md` = evidence index

## 6. Historical Docs

Historical docs 不应被改写成最新状态，而应有清晰 boundary marker。

历史文档保留历史事实；当前运行合同以 current docs、code、tests、schema registry、manifest.display 为准。

## 7. Code-Map as Current Implementation Learning Layer

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

它不负责：

- governance
- roadmap
- future architecture
- historical closeout
- project status
- research narrative
- completed milestone history

---

# Part 10 — Codex Instruction Layer

## 1. Instruction Hierarchy

优先级：

```text
1. Human directive / approved envelope
2. AGENTS.md
3. project skill
4. task template
5. current brief
6. current code/tests/contracts
7. full governance reference
8. future roadmap
```

## 2. AGENTS.md Role

`AGENTS.md` 是 AI execution constitution，负责：

- current phase
- hard out-of-scope
- private data boundaries
- forbidden actions
- stop conditions
- environment and path rules
- stable execution boundaries

## 3. Skill Role

Project skill 负责：

- workflow
- task modes
- review gates
- context routing
- contract maintenance playbooks
- verification profiles

Skill 不应 redefine architecture 或 override AGENTS。

## 4. Templates Role

Templates 负责：

- task-specific prompt structure
- recommended reasoning effort
- context budget
- allowed / forbidden placeholders
- checks
- output format

Templates 是日常任务入口，不应重复完整 governance theory。

## 5. Current Brief Role

Current brief 负责短状态：

- current phase
- current main status
- latest milestones
- current next step
- known limitations
- active docs pointers

它不替代 PROJECT_STATUS、README、code、tests 或 contracts。

## 6. Branch / Commit / Push Policy

基本原则：

- code / tests / contract-adjacent work 使用 semantic branch
- small docs-only cleanup 可在 clean main 上运行
- commit 后应 push 对应 branch
- merge 后应 push main
- no force push / rebase / amend / squash / merge commit unless explicitly approved
- output full SHA for confirmed git state

本地 confirmed Git 状态应同步 GitHub，方便后续 GPT / Codex / Pro 分析。

## 7. Subagent Policy

Subagents 只在 approved envelope 内使用。

规则：

- narrow context packet
- disjoint scope
- review-only subagents do not modify files
- shared contracts owned by main agent or explicit integrator
- subagents do not read full repo by default
- final report lists subagents used

Goal mode 可使用多个 subagents；ordinary tasks 默认不用。

---

# Part 11 — Review Checklist System

## 1. Review Layers

Review 至少包含：

- scope review
- contract review
- private-data review
- test review
- docs/code-map review when relevant

## 2. Scope Review

检查：

- changed files match allowed scope
- no unrelated refactor
- no hidden dependency
- no phase expansion
- no generated/private artifacts

## 3. Contract Review

检查：

- schema compatibility
- manifest.display coverage
- API payload stability
- frontend shared type alignment
- loader/writer consistency
- run identity consistency

## 4. Blast Radius Review

检查：

- affected modules
- affected tests
- affected docs
- affected generated artifacts
- affected frontend routes / state
- affected API/download/cache behavior

## 5. Half Migration Detection

Red flags：

- writer updated but loader not updated
- API updated but frontend still old
- schema changed but tests not changed
- docs changed but code disagrees
- display updated but download missing
- generated runs kept visible by weakening validation

## 6. Private Boundary Review

检查：

- no row-level private data
- no complete private path
- no credentials
- no local drive/user path
- no private generated artifact committed
- no public metadata containing local runtime state

## 7. Token / Context Review

对于 Codex workflow change，检查：

- ordinary tasks do not broad-load repo
- full standard not default daily context
- goal mode high token allowed only in approved envelope
- templates include context budget
- subagent scope is narrow

---

# Part 12 — ADR and Architecture Decision System

## 1. ADR Required For

ADR 或等价 architecture decision 适用于：

- new architecture
- phase upgrade
- new market/frequency model
- simulation domain introduction
- deployment/runtime topology
- schema-breaking migration
- major refactor
- new dependency/framework

## 2. ADR Separation Rule

ADR 记录 decision，不记录一次性 implementation log。Implementation details 应在 PR/commit/task summary 中。

## 3. ADR Structure

```text
Title
Status
Context
Decision
Alternatives
Consequences
Non-goals
Migration / rollback
Approval
```

## 4. Future Capability Admission Rule

Future capability 进入当前实现前必须证明：

- current phase needs it
- data boundary is clear
- contract is explicit
- tests exist or are planned
- complexity is justified
- non-goals are clear

## 5. Evolution Over Revolution

优先：

```text
small stable evolution
```

而不是：

```text
large speculative rewrite
```

---

# Part 13 — Practical Operating Rules

## 1. Before Starting a Task

确认：

```text
Task mode
Reasoning effort
Branch / worktree plan
Allowed scope
Forbidden scope
Context budget
Stop conditions
Checks
Commit / push policy
```

## 2. During a Task

规则：

- read only relevant context
- use rg before full reads
- update tests with behavior changes
- stop on unclear blast radius
- report rather than hide constraint conflicts
- avoid large uncontrolled output

## 3. Before Commit

检查：

- `git diff --name-only`
- relevant tests / static checks
- `git diff --check`
- private path / generated artifact search if relevant
- scope check
- docs sync if needed

## 4. After Commit / Merge

规则：

- push confirmed branch/main
- output full SHA
- verify remote when needed
- keep feature branches until user approves deletion
- do not silently amend or rewrite history

## 5. When To Spend High Token

High token is justified for：

- schema / manifest.display / API/frontend contract migration
- N6 prediction handoff design
- full N5 capability boundary decision
- Goal / Overnight implementation
- final cold review of broad contract work
- private-data boundary review

High token is not justified for：

- push / merge
- docs-only small sync
- git status / log / diff name-only
- simple branch ops
- repeated reading of unchanged docs

## 6. Standard Maintenance

This standard should evolve only when repeated workflow problems show that existing AGENTS / skill / templates are insufficient.

Do not update this standard for every task. Prefer:

1. tests for enforceable behavior
2. templates for task execution
3. skill for repeated workflow
4. AGENTS for hard boundaries
5. this standard for durable principles

---

# Appendix — Minimal Prompt Pattern

For ordinary tasks, the prompt should contain only task deltas:

```text
Task mode:
Recommended reasoning effort:
Goal:
Current branch / commit:
Allowed scope:
Forbidden scope:
Context budget:
Checks:
Output:
```

Do not paste the full standard, full AGENTS, full skill, full roadmap, or broad repo context into ordinary prompts.

For Goal / Overnight tasks, use an approved envelope instead of an unbounded prompt.
