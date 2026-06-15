# token 节约规则（Token economy）

本文件是硬规则，不是风格建议。目标是让用户用一两句话也能让 Codex 精准命中任务，少读、少问、少返工。

## 1. 任务上下文预算

| Mode ID | 默认预算 | 读取规则 |
|---|---|---|
| `small-fix` | small | `AGENTS.md` + 目标文件 + 最小测试；不读 `docs/architecture` |
| `standard-implementation` | medium | 先 owner layer；只读相关代码/tests/docs；不读全部 repo |
| `independent-review` | small/medium | 先 diff/stat；只在发现 issue family 后扩展 |
| `handoff-packaging` | small | 读 `.run` summary/package rules；不读业务代码 |
| `merge-gate` | small | 读 git status/log/compare；不读文件内容，除非 diff 异常 |
| `docs-status-sync` | small | 只读 active docs/status/brief/code-map 相关文件 |
| `governance-redesign` | medium/large | 读治理文件、evidence summary、必要代码事实；不读 private data 或旧 raw handoff |
| `apply-only-validation` | small | 读 patch summary、changed files、验证脚本；不重新设计 |

升级预算前必须说明：为什么小预算不够、将额外读取哪些文件、如何防止 scope creep。

## 2. Diff-first / file-second / repo-last

1. 先运行 `git status --short --branch`、`git diff --name-only`、`git diff --stat`。
2. 读 changed files 或目标文件。
3. 用 `rg -n` 找 issue family 的命中摘要。
4. 只打开最相关的少数文件。
5. 最后才扩大到 repo 级搜索。

## 3. 一次只读一个主模板

短提示词先读 `auto-task-router.md`；路由后最多读一个 primary template。不要一次读取 `references/templates/*.md` 全部文件。

## 4. Same issue family scan（同族问题扫描）

先用 `rg -n "pattern"` 看文件名和命中行，不要直接打开全部命中文件。命中很多时，先汇总 pattern 和 owner layer，最多打开 top 3-5 个最相关文件。

## 5. Evidence summary first（证据先读摘要）

`.run` 和 handoff evidence 先读 `summary.md`、`manifest.json`、`index.json`、package rules。不要默认读 raw logs、large stdout、parquet/csv/pkl/raw vendor data 或旧 handoff zip/json。

## 6. 脚本输出要短

默认 final report 只贴一行摘要和报告路径，不粘贴完整 stdout。失败时只贴定位所需的错误行。脚本可使用 `--json` 产出机器可读摘要。

## 7. 外部工具 token gate

先用 repo 内事实源。只有以下情况才用 Context7/GitHub MCP、plugins、web 或其他外部工具：用户明确要求；官方工具行为可能变化；repo 内没有答案且外部工具能减少返工。Binance plugin 默认不用于 A-share governance 任务。

## 8. Subagent token gate

默认不用 subagents。若使用，必须指定单一问题、只读范围、输出上限，并在 main agent 汇总后关闭；禁止让 subagent “全面看看”。

## 9. Final report length budget

- `small-fix`：3-6 行。
- `standard-implementation`：changed files + checks + risks。
- `independent-review`：findings table + evidence。
- `handoff-packaging`：package path + manifest + safety result。
- `goal-closure` / `governance-redesign` / `release-gate`：允许较详细，但仍不粘贴长日志。
