# 短提示词示例（Short prompt examples）

本文件用于验证“用户输入很少也能精准命中任务”的硬目标。示例不是固定话术，而是路由合同：短提示词应命中正确 mode、读取最小上下文、少问但不越权。

| 用户短提示词 | 应命中 mode | 应读上下文 | 禁止动作 | 是否追问 |
|---|---|---|---|---|
| 检查当前分支有没有 public API 扩张，只读 review。 | `independent-review` | `AGENTS.md`、diff/stat、Public Surface Gate、相关 changed files | 改代码、commit/push、读 private data | 不追问，除非 base 不清 |
| 同步 README 和 PROJECT_STATUS 的状态，只改文档，验证后提交。 | `docs-status-sync` | active docs/status、current brief、docs template | 改 `src/**`、改业务 tests | 分支/commit 权限不清时追问 |
| 按 `.run` 官方脚本生成 handoff 包，保持 public-safe。 | `handoff-packaging` | `.run` summary/package rules、evidence-and-handoff、package safety script | 手工伪造包、包含 private rows | `.run` 工具不存在时报告 `BLOCKED` |
| 应用这个 patch，验证没问题就 commit/push，不要重新设计。 | `apply-only-validation` | patch、apply template、changed files、validation scripts | 重新设计、顺手改业务代码 | patch 冲突时停止 |
| 修复 factor catalog 里的一个 bug，只改归属层。 | `standard-implementation` | repo-map、owner routing、相关 code/tests | 跨层重构、public surface 扩张 | bug 目标不清时追问 |
| 把这个 PR 做 release gate。 | `release-gate` | branch/compare、active docs、public/private gates、test evidence | 顺手修业务代码 | release base/权限不清时追问 |
| 继续做 N3。 | `ambiguous` | 只读 `AGENTS.md` 和 current status | 直接实现 | 必须追问 |
| 看看有没有问题。 | `ambiguous` 或 `independent-review` | 如果有当前分支，则先 diff/stat；否则问 scope | 写代码 | scope 不清时追问 |

## 判断原则

1. 短提示词优先自动路由，不要把每个细节都反问给用户。
2. 只有 blocking ambiguity 才追问：分支/base、写权限、private data、public API/CLI、merge/push/release、用户改动归属。
3. 非阻塞细节用 mode 默认值。
4. 短提示词不等于低安全；所有 stop conditions 仍然有效。
