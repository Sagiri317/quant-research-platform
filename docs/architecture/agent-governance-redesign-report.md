# Agent Governance Redesign Report

本文记录本轮 Agent Governance Redesign（智能体治理重构）的诊断、目标架构、能力取舍、文件决策、验证清单和迁移计划。它是 architecture reference（架构参考），不是日常任务 prompt；日常任务从 `AGENTS.md` 和 `.agents/skills/quant-research-platform-maintenance/SKILL.md` 进入。

## 1. Governance diagnosis / 治理诊断

### 1.1 当前主要问题

| 问题 | 证据来源 | 影响 | 治理修复 |
|---|---|---|---|
| 任务类型混淆 | conversation evidence 显示 handoff/package/governance/N2 acceptance 容易混用 | Codex 可能做错任务、重复审 N2、或把治理任务误做成业务实现 | `Task Router Gate` + `task-modes.md` + router template |
| Artifact truthfulness（产物真实性）风险 | `.run` evidence 中存在 `PASS/FAIL/NOT_RUN` 敏感项 | 未跑检查可能被写成 PASS，FAIL 可能被掩盖 | `contracts-and-verification.md` + Artifact Truth Gate |
| `.codex` 与 `.agents` 路径混淆 | stale/conflicting guidance scan、global `.codex` 摘要 | 项目治理可能被迁移到错误层，或把 global raw logs 当 repo 文件 | `AGENTS.md` path authority + `agent-governance.md` |
| `.run` 工具未被稳定路由 | `.run` 是 ignored workflow/evidence layer（本地证据层） | handoff 可能反向驱动架构；package safety 依赖人工记忆 | handoff template + package safety script |
| ps1-only 脚本风险 | script language scan 发现 `.agents` 内 PowerShell-only 脚本 | macOS/Linux/WSL 不可运行，Win11 以外返工 | 删除 ps1-only，新增 Python stdlib scripts |
| stale status（状态漂移） | active status/brief 中保留旧分支/旧 next step | 后续 agent 读取错误 current truth | 重写 `PROJECT_STATUS.md` 与 current brief |
| public surface expansion 风险 | scans 命中 registry/store/facade/Repository/future-only terms | 小任务可能无意新增 public API/CLI/registry/store/facade | Public Surface Gate + owner-layer routing |
| token 浪费 | evidence 指出重复读旧 handoff、全 repo、全模板 | quota 浪费且上下文噪声增加 | progressive disclosure（渐进式加载）+ context budget |

### 1.2 返工根因分类

1. Mode ambiguity（模式歧义）：merge、handoff、review、implementation、governance 没有先分流。
2. Evidence ambiguity（证据歧义）：official compare、smoke、full-horizon、NOT_RUN 没有统一状态合同。
3. Path authority ambiguity（路径权威歧义）：global `.codex`、repo `.agents`、repo-local `.codex` 角色不清。
4. Owner ambiguity（归属歧义）：P2/P3 和 issue family 没有 owner layer gate。
5. Portability ambiguity（平台歧义）：PowerShell-only 脚本进入治理层。
6. Documentation truth drift（文档事实漂移）：旧 branch/status/next-step 写入 active docs。

### 1.3 token 浪费来源

- 把旧 N2 handoff 当成本轮主要输入。
- 短任务读取全部 templates、docs、history、`.run` 原始输出。
- review 后不把同类问题转成 gates/scripts，下轮重复扫描。
- 把 `.run` handoff 结构当长期架构上下文。
- 通过长 `AGENTS.md` 承载所有知识，而不是分层按需读取。

## 2. Target governance architecture / 目标治理架构

| 层 | action | owner | 读取时机 | token 预算 | 职责 |
|---|---|---|---|---|---|
| `AGENTS.md` | rewrite | project governance | 每次任务 | 小，目标 <12KB | hard boundaries、task router、path authority、stop conditions |
| `.agents/briefs/current-qrp-status.md` | rewrite | current truth | 当前事实需要时 | 小 | baseline、public surface、private boundaries |
| `.agents/briefs/user-principles.md` | add | user principle layer | 架构/命名/public surface/cross-layer 任务 | 小 | 长期偏好和不可违反原则 |
| `SKILL.md` | rewrite | operational workflow | QRP maintenance skill 触发 | 小 | load order、task modes、truth rules |
| `references/*.md` | add/rewrite/merge | governance references | 按 task mode 加载 | 小/中，分文件 | task modes、context routing、owner routing、contracts、gates、handoff、portability |
| `templates/*.md` | merge/rewrite | execution checklist | 只加载匹配 mode 的模板 | 小 | mode-specific execution，不做百科 |
| `scripts/*.py` | add/replace | deterministic gates | validation 阶段执行 | 0 context when executed | governance/package/code-map 检查，Python stdlib |
| `docs/architecture/agent-governance*.md` | add | durable rationale | governance/architecture review | 不日常读 | diagnosis、target design、draft adapters |
| `.run` local layer | keep ignored | workflow evidence | handoff/evidence 任务 | 不默认读 | package/evidence tooling，不作为架构源 |
| hooks/MCP/plugins/subagent configs | draft-only | optional adapter | 另行批准后 | 默认 0 | future adapters，不默认启用 |

设计目标是：更短入口、更强 gates、更小 context、更明确 task mode、更强 public-safe/private-safe 边界、更真实 final/handoff claims。

## 3. Capability decision matrix / 能力取舍

| Capability | QRP stance | 减少的返工 | 成本/风险 | 落点 |
|---|---|---|---|---|
| `AGENTS.md` | use now | 硬边界、stop conditions、path authority | 过长会浪费 token | short root file |
| nested/directory AGENTS | later only | 子目录 owner rules | 多层指令叠加复杂 | 暂不新增 |
| repo `.agents/skills` | use now | progressive disclosure、任务流程封装 | description 不清会误触发 | `.agents/skills/...` |
| skill descriptions | use now | 更准触发 skill | 过宽会误触发 | `SKILL.md` frontmatter |
| templates/prompts | use now | 标准化 task modes | 模板过多会冲突 | merged templates |
| skill scripts | use now | 机械检查 private/package/path/portability | 需跨平台维护 | Python stdlib scripts |
| hooks | draft-only | 可阻断危险操作 | 需要 trust、repo-local `.codex`、误拦截 | adapter doc |
| deterministic guard scripts | use now | stale status/private package/public path 漏检 | 覆盖有限 | scripts/tests |
| subagents/reviewer profiles | controlled use | 大审查按 issue family 隔离 | token 成本高、integration 风险 | template policy |
| MCP | draft-only | 外部动态上下文 | 凭据/网络/权限/维护成本 | adapter doc |
| plugins | draft-only | 多 repo 分发 | manifest/versioning/trust 成本 | adapter doc |
| Codex CLI config/profiles | optional manual only | 本机行为稳定 | 不应项目默认修改 global `.codex` | docs only |
| approval/sandbox/network policy | document only | 避免危险运行 | 与用户本机配置相关 | AGENTS/templates |
| Codex CLI/IDE/cloud/app | support through neutral rules | 多入口一致 | 工具能力差异 | agent-neutral docs |
| non-interactive `codex exec` | later/draft | CI-style repeat checks | auth/CI/permissions | adapter doc |
| GitHub integration/PR review | later/draft | PR review 自动化 | secrets/费用/权限 | adapter doc |
| memory/summaries/repo briefs | use summaries | 减少重复解释 | stale 风险 | `.agents/briefs` |
| recurring automation/scheduled stale checks | later | 定期 docs drift 检查 | false positives | future issue |
| handoff packaging scripts | use via `.run` when requested | 包安全和 truthfulness | `.run` ignored；需检查存在 | handoff template |
| review bundle/git bundle workflow | use for review handoff | reviewer 可复现 | 包安全和体积 | `.run` + safety script |
| public-safe metadata index | later | 检索/知识图谱 | 容易变成新 store/facade | future design |
| `.run` workflow evidence | use as evidence | 审查闭环 | 不可反向驱动架构 | handoff only |
| conversation logs evidence | summarized extracts only | 提取踩坑模式 | 不完整，不能编造 | diagnosis only |

## 4. File decision table / 文件决策表

| Action | Path | Purpose | Loaded by/when | Reduces rework | Saves token | Risk | Confidence |
|---|---|---|---|---|---|---|---|
| rewrite | `AGENTS.md` | hard boundaries/router/path authority | every task | 阻断越界、路径混淆、false PASS | 短文件 | 规则过少需 references 承接 | high |
| rewrite | `README.md` | project entry + governance map | humans/agents | 快速找到治理入口 | 小 | README 仍需维护 | high |
| rewrite | `PROJECT_STATUS.md` | current operational truth | status/merge/docs tasks | 移除 stale branch/next step | 小 | 未来会过期 | high |
| rewrite | `.agents/briefs/current-qrp-status.md` | short current truth | QRP tasks | 防旧 N2 状态污染 | 小 | 需随阶段更新 | high |
| add | `.agents/briefs/user-principles.md` | stable user preferences | architecture/governance/public surface tasks | 减少反复解释长期原则 | 按需读 | 需保持短 | high |
| rewrite | `.agents/skills/quant-research-platform-maintenance/SKILL.md` | operational load order | QRP skill trigger | 统一 workflow 和 truth rules | progressive disclosure | trigger wording 需观察 | high |
| rewrite | `references/context-routing.md` | context budget | medium+ tasks | 防读错/读多 | 明确按需读 | 需要执行者遵守 | high |
| rewrite | `references/contracts-and-verification.md` | contract/truth rules | contract/artifact/API/private tasks | 防 `PASS/FAIL/NOT_RUN` 返工 | 集中规则 | 非完整 schema | high |
| add | `references/task-modes.md` | mode matrix/reasoning intensity | router | 防任务类型混淆 | 小表 | 需维护 | high |
| add | `references/owner-layer-routing.md` | owner/P2/P3 routing | implementation/review | 防跨层顺手修 | 小表 | owner map 需更新 | high |
| add | `references/governance-gates.md` | 16 gates | medium+ tasks | gate 标准化 | 按需读 | 机器检查有限 | high |
| add | `references/agent-capability-decision-matrix.md` | capability decisions | governance tasks | 防盲目启用 hooks/MCP/plugins | 按需读 | 功能变化需更新 | high |
| add | `references/evidence-and-handoff.md` | package/evidence truth rules | handoff/report tasks | 防 private package 和 false PASS | 按需读 | 需配合 `.run` | high |
| add | `references/governance-verification-checklist.md` | apply/review verification | after governance changes | 防漏验证 | 小 | 需执行 | high |
| add | `references/review-fix-loop.md` | issue-family review loop | goal/review tasks | 防同族问题漏修 | 小 | 依赖 reviewer 质量 | high |
| add | `references/script-portability.md` | script compatibility rules | script changes | 防 Win/mac/Linux 返工 | 小 | 不覆盖业务脚本 | high |
| add/rewrite | `references/templates/README.md` | template index | template selection | 防加载全部模板 | 小 | 需随模板更新 | high |
| rewrite | `templates/auto-task-router.md` | short prompt routing | short/ambiguous prompts | 防任务误路由 | 小 | 仍需人工判断 | high |
| rewrite | `templates/focused-implementation.md` | bounded implementation | standard implementation | 防 scope creep | 小 | 需要 owner files | high |
| rewrite | `templates/focused-audit.md` | review-only | independent review | 防 review 变实现 | 小 | 需要清晰 scope | high |
| add | `templates/goal-closure.md` | long goal/review-fix loop | approved goal closure | 防多轮返工 | 小 | token 成本仍高 | high |
| add | `templates/handoff-packaging.md` | handoff package workflow | handoff tasks | 防包安全/manifest 漏项 | 小 | `.run` 工具需本地存在 | high |
| add | `templates/git-operations.md` | branch/commit/merge gates | git tasks | 防误 merge/push/main | 小 | 需要用户批准 | high |
| add | `templates/docs-status-sync.md` | active docs sync | docs/status tasks | 防 stale status | 小 | 可能漏 archive docs | high |
| add | `templates/apply-only-validation.md` | apply patch/zip exactly | apply-only tasks | 防二次 redesign | 小 | patch conflict 时需停 | high |
| delete | old narrow template aliases | remove parallel rules | n/a | 防旧规则保护旧结构 | 少加载 | 外部旧 prompt 需更新 | medium-high |
| delete | `agents/openai.yaml` | remove Codex-specific pseudo-profile | n/a | 保持 agent-neutral | 少路径 | 若外部引用需更新 | medium |
| delete | `references/code-map-flow.md` | remove old code-map reference | n/a | 防旧 ps1 flow | 少路径 | 已由 Python script/checklist 承接 | medium |
| delete | `scripts/*.ps1` | remove ps1-only scripts | n/a | 防 macOS/Linux/WSL 失败 | 少路径 | 手动用户需改 Python | high |
| add | `scripts/verify_governance.py` | deterministic governance gate | validation | 防 business code/.codex/stale/path/script issues | 执行代替阅读 | 覆盖有限 | high |
| add | `scripts/package_safety_check.py` | package safety scanner | handoff packaging | 防 blocked files/private paths/local paths | 执行代替人工 | 内容扫描保守 | high |
| add | `scripts/verify_code_map_flow.py` | cross-platform code-map anchor check | code-map/governance verification | 替代 ps1-only code-map check | 执行代替人工 | 只检查 anchors | medium-high |
| add | `docs/architecture/agent-governance.md` | durable target architecture | governance/architecture review | 防规则散落 | 不日常读 | 需避免变成长 prompt | high |
| add | `docs/architecture/agent-governance-adapters-draft.md` | optional adapters | future adapter research | 防默认启用 hooks/MCP/subagents | 不日常读 | 可能过期 | high |
| add | `docs/architecture/agent-governance-redesign-report.md` | diagnosis/design/migration | governance review | 保留本轮研究结论 | 不日常读 | 较长 | high |
| update | `docs/architecture/README.md` | architecture reading order | architecture tasks | 可发现 governance docs | 小 | 无 | high |
| update | `docs/architecture/open-issues-register.md` | governance follow-ups | architecture/status tasks | P2/P3 路由清晰 | 小 | 表增长 | medium |

## 5. Draft-only files / 暂不启用内容

`docs/architecture/agent-governance-adapters-draft.md` 记录以下 draft-only（草案）能力：repo-local `.codex` hooks/config、MCP、subagents/reviewer profiles、plugins、Claude-specific adapters。当前不建议落地；若未来落地，必须独立分支、无业务代码改动、跨平台 dry-run/check-only 验证、可删除回滚。

## 6. Governance verification checklist / 验证清单

Codex apply-only 后应检查：

- markdown formatting：无坏 code fence、尾随空格、明显坏标题。
- broken relative paths：新文档引用的 repo paths 存在或明确是 draft/future。
- stale N1/N2/N3 status：active docs 不再把旧 branch 写成当前分支/next step。
- oversized `AGENTS.md`：仍保持短小，不写成长篇百科。
- private data：无 private rows、vendor markers、stock-code samples、local absolute paths、credentials、tokens。
- business code changes：`src/**`、business `tests/**`、`pyproject.toml`、`environment.yml` 未改。
- public API changes：未新增 API/CLI/registry/store/facade。
- contradictory instructions：`.agents` 与 `AGENTS.md` 对 `.codex/.agents/.run` 边界一致。
- skill trigger clarity：`SKILL.md` description 有明确 use/do-not-use。
- context routing clarity：short prompt 能路由到一个模板。
- script portability：Python stdlib，Win11/macOS/Linux/WSL 路径安全，无 ps1-only。
- handoff packaging：`.run` 保持 ignored workflow layer；package safety script 可扫 zip/dir。

Recommended commands:

```text
git status --short --branch
git diff --name-only
git diff --check
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --repo-root . --governance-only
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python .agents/skills/quant-research-platform-maintenance/scripts/package_safety_check.py --help
```

## 7. Migration plan / 迁移计划

### Phase 1: immediately apply

1. 从 `feature/qrp-clean-rewrite` 新建 `feature/agent-governance-redesign`。
2. 应用 `governance-redesign.patch` 或 zip 中 repo-relative files。
3. 运行 governance-only 验证、code-map flow check、package safety help/self-check、`git diff --check`。
4. 确认无 `src/**`、business tests、dependency/env 改动。
5. 生成新的 public-safe handoff/review package 时，使用 `.run` 官方工具并额外用 package safety script 扫描。

### Phase 2: after local testing

1. 在用户 Win11 + Python 3.13.14 环境运行脚本。
2. 用 fake handoff zip fixture 测试 `package_safety_check.py` 的 PASS/FAIL 行为。
3. 用 mock governance diff 测试 `verify_governance.py --governance-only`。
4. 更新外部 Codex/GPT prompt templates，删除旧 template path、旧 ps1 script、旧 `.codex` workflow 引用。

### Phase 3: optional future enhancements

1. 若 handoff 返工仍多，考虑 tracked wrapper 指向 `.run` handoff tool，但不把 `.run` 变成架构源。
2. 若危险操作反复出现，在 sandbox 中测试 repo-local `.codex` hooks draft。
3. 若外部动态上下文变多，再评估 MCP。
4. 若多 repo 复用治理层，再研究 plugin packaging。
5. 若大型 review token 成本仍高，再定义 read-only subagent profiles。
