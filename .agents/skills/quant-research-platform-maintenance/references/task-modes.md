# Task Modes / 任务模式

任务开始时先选 mode，再选模板。不要让 short prompt 自动升级成跨层实现。

| Mode | 适用场景 | 默认权限 | 必读上下文 | 默认验证 |
|---|---|---|---|---|
| `small-fix` | 单文件或小范围 typo/docs/test expectation 修复 | 小范围改动；不改 public surface | `AGENTS.md` + 目标文件 | `git diff --check` + 相关最小检查 |
| `standard-implementation` | 有明确 owner layer 的功能/bugfix | 只改 owner layer 和必要 tests/docs | current status、context routing、owner-layer routing、相关代码/tests | targeted pytest + ruff + diff check |
| `goal-closure` | 长任务、review-fix loop、overnight goal | 仅在用户批准 envelope 内闭环 | task envelope、task modes、gates、owner routing、relevant code/tests | 每 slice narrow tests；最终 full relevant gates |
| `independent-review` | 只审查、不改代码 | read-only；可输出 findings | diff/branch/目标文件；必要时 issue-family references | review checklist；不得写 PASS 给未跑检查 |
| `handoff-packaging` | 生成 review bundle / handoff package | 使用 `.run` 官方工具；不改架构 | `.run` 工具说明、package rules、truth rules | package safety + verify package + artifact truth |
| `merge-gate` | ff merge、release gate、branch gate | 只做 Git gate；不做业务修复 | branch status、diff、review evidence | git status/log/merge-base；禁止 force |
| `release-gate` | 发布/展示前检查 | check-only unless approved | README/PROJECT_STATUS/docs/api/docs/architecture | public surface + private safety + docs/status sync |
| `docs-status-sync` | current docs/status/brief/code-map 同步 | docs only；不改业务代码 | changed surface + active docs | markdown/path check + stale status scan |
| `governance-redesign` | 改 AGENTS/.agents/skill/scripts/gov docs | governance-only；不改 `src/**`/business tests | governance files + evidence pack | governance script + package safety + docs consistency |
| `apply-only-validation` | 应用外部 patch/zip，不重新设计 | apply exactly; validate only | patch/zip + apply prompt + changed files | governance script + listed checks |
| `ambiguous` | 目标/权限/路径不清 | no edit | ask user | none |

## Reasoning intensity / 推理强度建议

- Low：branch push、commit existing changes、single-doc typo。
- Medium：small-fix、docs-status-sync、focused audit。
- High：standard implementation、public-surface-adjacent changes、private-safety-adjacent changes。
- Highest available：architecture closure、governance redesign、goal closure、release gate、multi-layer review。

## Subagent policy / 子智能体策略

- 默认不使用 subagents。
- 使用时按 issue family 分配，不按“随便复查”分配。
- review subagent 默认 read-only。
- main agent 负责 integration、truthfulness、commit、push、handoff。
- subagent 消耗额外 token；只有当它减少返工或隔离复杂审查上下文时才使用。
