# 治理文件人话说明（Agent governance file guide）

本文用人话说明 QRP 治理体系每个目录和文件是干什么的、谁会读、什么时候读、怎么维护。

## 根目录文件

- `AGENTS.md`：所有 AI agents 的“开工守则”。它告诉 Codex 这个项目是什么、不是什么、哪些事不能做、短提示词怎么路由、什么时候必须停下来问。它必须短，不能写成百科。
- `README.md`：给人和 agent 的项目入口。说明 QRP 当前定位、公共接口、私有数据边界和治理入口。
- `PROJECT_STATUS.md`：当前状态板。记录当前分支、阶段、下一步和 blocker。它容易过期，所以每次 docs/status sync 都要检查。

## `.agents/briefs/`

- `current-qrp-status.md`：当前事实短简报。给普通任务快速了解当前 public surface、owner layer、私有边界、资源事实。
- `repo-map.md`：很短的仓库路由地图。它不是完整 code-map，只帮助 Codex 少读文件、先定位 owner layer。
- `user-principles.md`：用户长期偏好。比如语义收口、命名一等、替代优先、80% 内存预算事实源、短提示词和省 token 是硬目标。

## `.agents/skills/quant-research-platform-maintenance/`

- `SKILL.md`：QRP 维护 skill 的入口。它规定加载顺序、默认语言、任务模式、真实性规则和脚本入口。

### `references/`

- `task-modes.md`：任务模式表。把任务分成 `small-fix`、`standard-implementation`、`independent-review` 等，防止短提示词误路由。
- `token-economy.md`：token 节约规则。规定 diff-first、一次只读一个模板、summary-first、短 final report 等硬规则。
- `context-routing.md`：上下文路由规则。告诉 Codex 不同任务先读什么、不要读什么。
- `governance-gates.md`：一遍过 gates 清单。把“不要返工”落成具体检查门。
- `owner-layer-routing.md`：owner layer 路由表。防止一个小任务跨层乱改。
- `contracts-and-verification.md`：公共接口、artifact truth、private safety 和验证阶梯。
- `evidence-and-handoff.md`：`.run`、handoff package、handoff evidence、Git bundle 的证据规则；review bundle 是 historical / migrated 旧称。
- `review-fix-loop.md`：review 后怎么按 issue family 修，避免只修一个点。
- `script-portability.md`：治理脚本跨 Win11/macOS/Linux/WSL 的规则。
- `governance-verification-checklist.md`：治理变更后的验证清单。
- `external-tools-and-adapters.md`：Context7/GitHub MCP、Binance plugin、superpowers skills、hooks、subagents、GitHub Action 等如何与 QRP 共存。
- `agent-capability-decision-matrix.md`：各种 Codex/agent 能力当前用不用、为什么、落在哪一层。

### `references/templates/`

这里是执行模板。普通任务只读一个最匹配的模板，不要全读。

- `auto-task-router.md`：用户只写一两句话时，先判断 task mode。
- `focused-implementation.md`：有明确 owner layer 的实现任务。
- `focused-audit.md`：只读审查任务。
- `goal-closure.md`：长目标和 review-fix loop。
- `handoff-packaging.md`：生成 public-safe handoff package。
- `git-operations.md`：commit、push、merge、branch cleanup。
- `docs-status-sync.md`：同步 README、PROJECT_STATUS、briefs 等 active docs。
- `apply-only-validation.md`：精确应用 patch/zip，只验证不重新设计。

### `scripts/`

- `verify_governance.py`：检查治理层变更有没有碰业务代码、private roots、repo-local `.codex`、blocked extensions、stale status、明显敏感内容。
- `package_safety_check.py`：扫描 handoff package，防止包含 private data、blocked extensions、secret-like 内容、本地路径。
- `verify_code_map_flow.py`：检查 `docs/code-map/platform-flow.md` 是否仍包含当前 clean rewrite 的关键锚点。

## `docs/architecture/`

- `agent-governance.md`：治理架构总说明。
- `agent-governance-redesign-report.md`：本轮重构诊断、目标、能力取舍和迁移计划。
- `agent-governance-adapters-draft.md`：MCP、hooks、plugins、subagents、codex exec、GitHub Action 的 draft-only 方案。
- `agent-governance-file-guide.md`：本文。用于人快速理解每个治理文件。
- `README.md`：架构文档目录入口。
- `open-issues-register.md`：开放问题登记；治理相关 follow-up 要登记在这里。

## `.run/`

`.run/` 不作为项目治理根。它是本地 evidence/tooling 层：生成 handoff 包、保存验证摘要、记录 public-safe evidence。日常任务不要默认读 `.run` raw logs。
