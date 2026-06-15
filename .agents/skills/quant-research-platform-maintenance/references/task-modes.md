# 任务模式（Task modes）

任务开始时先选 mode ID，再选模板。短提示词不能自动升级成跨层实现；长提示词也不能绕过 gates。

| Mode ID | 中文名 | 适用场景 | 默认权限 | 必读上下文 | 默认验证 |
|---|---|---|---|---|---|
| `small-fix` | 小修 | 单文件 typo、轻量 docs、窄测试 expectation 修复 | 小范围改动；不改 public surface | `AGENTS.md` + 目标文件 | `git diff --check` + 最小相关检查 |
| `standard-implementation` | 标准实现 | 有明确 owner layer 的功能或 bugfix | 只改 owner layer 和必要 tests/docs | current status、repo-map、context routing、owner routing、相关代码/tests | targeted pytest + ruff/py_compile + diff check |
| `goal-closure` | 目标闭环 | 长任务、review-fix loop、overnight goal | 只在用户批准 envelope 内闭环 | task envelope、task modes、gates、owner routing、相关代码/tests | 每 slice narrow tests；最终 full relevant gates |
| `independent-review` | 独立审查 | 只审查、不改代码 | read-only；输出 findings | diff/branch/目标文件；必要时 issue-family references | review checklist；未运行不得写 PASS |
| `handoff-packaging` | 交接打包 | 生成 review bundle / handoff package | 使用 `.run` 官方工具；不改架构 | `.run` summary/package rules、evidence rules、package safety | package safety + manifest truth + artifact truth |
| `merge-gate` | 合并门 | ff merge、branch gate、release merge | 只做 Git gate；不做业务修复 | branch status、diff、review evidence | git status/log/merge-base；禁止 force |
| `release-gate` | 发布门 | 发布/展示前检查 | check-only unless approved | README/PROJECT_STATUS/docs/api/docs/architecture | public surface + private safety + docs/status sync |
| `docs-status-sync` | 文档状态同步 | active docs/status/brief/code-map 同步 | docs only；不改业务代码 | changed surface + active docs | markdown/path check + stale status scan |
| `governance-redesign` | 治理重构 | 改 `AGENTS.md`、`.agents/`、skill、scripts、governance docs | governance-only；不改 `src/**`/business tests | governance files + evidence summary + code facts | governance script + package safety + docs consistency |
| `apply-only-validation` | 补丁应用验证 | 应用外部 patch/zip，不重新设计 | apply exactly; validate only | patch/zip + apply prompt + changed files | governance script + listed checks |
| `ambiguous` | 歧义任务 | 目标/权限/路径不清 | no edit | ask only if blocking | none |

## 推理强度建议

- Low：branch push、commit existing changes、single-doc typo。
- Medium：small-fix、docs-status-sync、focused audit。
- High：standard-implementation、public-surface-adjacent changes、private-safety-adjacent changes。
- Highest available：architecture closure、governance-redesign、goal-closure、release-gate、multi-layer review。

## blocking ambiguity（必须追问）

只有这些情况必须问：分支/base 不清、是否允许写操作不清、是否允许 private data 不清、是否允许 public API/CLI 扩张不清、是否 merge/push/release 不清、用户已有改动归属不清。

命名细节、默认验证范围、docs 是否同步这类非阻塞问题，应按 mode 默认规则执行，不要浪费一轮追问。

## Subagent policy（子智能体策略）

- 默认不用 subagents。
- 只有复杂 read-only review、release-gate、goal-closure 后期审查、governance-redesign 审查可考虑。
- 每个 subagent 只负责一个 issue family，必须有只读范围和输出上限。
- main agent 负责 integration、truthfulness、commit、push、handoff。
- subagent 消耗额外 token；只有能减少返工或隔离复杂上下文时才使用。
