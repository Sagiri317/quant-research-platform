# 外部工具与适配器（External tools and adapters）

本文件定义 QRP 治理与用户本地已有 Codex 能力的共存规则。QRP repo 不修改 global `.codex/`，不创建 repo-local `.codex/`，不默认启用 hooks、MCP、plugins、custom subagents。

## 总规则

外部工具可以辅助，但不能绕过 QRP 的 `AGENTS.md`、task modes、owner layer、public surface、private safety、artifact truthfulness。先用 repo 内事实源；只有明确需要时再用外部工具。

## 能力矩阵

| 能力 | 当前状态 | QRP 立场 | 可减少的返工 | 风险 | 落点 |
|---|---|---|---|---|---|
| Context7 MCP | 用户本地可能已有 | 可按需只读查官方/库文档 | 减少过期库知识 | 网络/噪声/token | 外部工具，不写 repo config |
| GitHub MCP | 用户本地可能已有 | 可查 PR/branch/commit；写操作受 Git gate 约束 | 减少手动查 GitHub | 权限/误写 | 外部工具，不绕过 merge gate |
| Binance plugin | 用户本地可能已有 | 当前 A-share governance 默认不用；未来 crypto/高频任务另行批准 | 未来数据研究 | 误触发、外部数据边界 | 禁止默认调用 |
| superpowers skills | 用户本地可能已有 | 可辅助方法论；不得覆盖 QRP 规则 | 规划/审查质量 | 抢触发、token | QRP rules 优先 |
| 其他 global skills | 用户本地可能已有 | 仅作辅助；QRP skill scope 要清楚 | 提效 | 触发冲突 | 不写 repo 配置 |
| hooks | Codex 支持 | draft-only；等脚本稳定后再启用 | 自动阻断危险操作 | trust/跨平台/误拦截 | docs only |
| custom subagents | Codex 支持 | 不创建 repo-local `.codex/agents`；可提示使用 built-in read-only subagents | 并行审查 | token 成本高 | template policy |
| MCP project config | Codex 支持 | 不创建 repo-local `.codex/config.toml` | 项目级工具配置 | 违反当前边界 | 禁止默认 |
| plugins | Codex 支持 | 不为 QRP 创建 plugin；未来跨 repo 分发再考虑 | 分发 skills | 权限/维护 | later |
| `codex exec` | Codex 支持 | draft-only automation examples | 可重复审查 | auth/CI 风险 | docs only |
| GitHub Action | Codex 支持 | later；CI/security review 后再考虑 | PR/release 自动审查 | secrets/runner 权限 | draft-only |

## 触发约束

QRP skill description 应聚焦 `quant-research-platform` 仓库内治理、维护、实现路由、private safety、handoff。不要写成 “all coding tasks”，避免和 superpowers/global skills 抢触发。

## 使用外部工具的条件

只在以下情况下使用外部工具：用户明确要求；官方工具行为可能变化；repo 内无答案且外部工具能减少返工；需要 GitHub PR/branch/commit 事实。使用后 final report 只写结论和关键证据，不粘贴长输出。
