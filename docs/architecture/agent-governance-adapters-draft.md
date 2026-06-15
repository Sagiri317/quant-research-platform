# 智能体治理适配器草案（Agent governance adapters draft）

本文只记录可选适配器设计，不启用任何 hooks、MCP、plugins、GitHub Action、custom subagent config，也不创建 repo-local `.codex/`。

## MCP

Codex 支持 MCP，用于连接第三方文档和工具。QRP 当前不写 `.codex/config.toml`。用户本地已有 Context7/GitHub MCP 时，可作为外部只读工具使用：Context7 查官方/库文档，GitHub MCP 查 PR/branch/commit。任何会读取 private data、vendor data、credentials、数据库、邮箱或浏览器私密上下文的 MCP 默认禁止。

## Hooks

Hooks 可在 Codex 生命周期中运行确定性脚本。QRP 当前不启用 repo-local hooks，因为它涉及 trust、跨平台、误拦截和本地环境差异。后续若启用，应先把对应逻辑沉淀为 Python standard library 脚本，并在 Win11/macOS/Linux 验证稳定。

## Subagents

Codex 支持内置 subagents，也支持 global 或 repo-local custom agents。QRP 当前不创建 `.codex/agents/`。可在 prompt 层要求 read-only subagents 做独立 review：例如 public surface、private safety、docs/status drift、same issue family。每个 subagent 必须单一问题、只读范围、输出上限。

## Plugins

Plugins 可打包 skills、apps、MCP servers。用户本地可能已有 Binance plugin、superpowers skills 或其他 global plugins。QRP 当前不创建 plugin；未来只有需要跨 repo/team 分发 QRP governance skill 时才考虑。

## `codex exec` 与 GitHub Action

`codex exec` 适合重复审查和 CI-style check；GitHub Action 可在 PR/release 中运行 Codex。当前只保留草案，不启用。任何 CI 自动化都必须经过 secrets、runner permissions、private data、patch artifact safety 审查。

## 回滚原则

所有 adapter 必须可关闭、可回滚、默认不影响 repo。任何示例文件必须是 `.example` 或文档片段，不能被 Codex 自动加载为活跃配置。
