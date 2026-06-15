# QRP 项目状态

本文件记录当前开发状态，不替代代码、tests、contracts 或 `AGENTS.md`。AI agents 执行任务前仍必须用 `git status --short --branch` 和当前分支 diff 校验事实。

## 当前分支与阶段

| 项目 | 当前事实 |
|---|---|
| 长期主分支 | `main` |
| 当前 clean baseline | `feature/qrp-clean-rewrite` |
| 治理重构状态 | 已 fast-forward merge 到 `feature/qrp-clean-rewrite` |
| 当前阶段 | Clean Rewrite `0.1.0`，Local Research Platform（本地研究平台） |
| 当前启用市场 | A-share（A 股） |
| 非目标 | paper trading、live trading、production trading、broker/exchange connector、deployment |

## 当前公共接口

```text
GET /api/0.1/catalog/factors
GET /api/0.1/catalog/labels
GET /api/0.1/research-datasets/{dataset_id}/requirements
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

当前 private local CLI（私有本地命令行）仅限：

```text
qrp-private schema-inventory --project-root . --output <path>
```

未经明确批准，不新增 public API、CLI、registry、store、facade、artifact lifecycle API、model registry、feature store、scheduler、worker、external API、broker/exchange connector、paper/live trading 入口。

## 当前治理状态

- 项目治理根是 `.agents/`。
- global `.codex/` 只作为用户显式提供的只读证据源。
- repo-local `.codex/`、hooks、MCP、plugins、subagent runtime config 默认不启用。
- `.run/` 是 ignored local workflow/evidence/tooling layer（本地工作流/证据/工具层），不是架构源。
- Agent governance redesign 已合入 clean baseline；后续治理变更应从 `feature/qrp-clean-rewrite` 新开分支或在明确批准的目标分支上小改。

## 当前 blocker / 风险

1. Public surface（公共接口面）必须保持最小，除非有单独批准。
2. 私有数据和私有报告不得进入 public-safe artifact 或 review/handoff 包。
3. 未运行检查必须标记为 `NOT_RUN` 或 `BLOCKED`，不能写成 `PASS`。
4. 后续 clean -> main 需要单独 merge gate / release gate 决策，不要直接碰 `main`。
5. 后续功能或治理修补默认从 `feature/qrp-clean-rewrite` 新开分支。

## 下一步

1. 对 `feature/qrp-clean-rewrite` 执行本地 Win11 + Python 3.13.14 治理验证。
2. 决定 clean -> main merge gate / release gate 是否启动。
3. 后续功能或治理修补从 `feature/qrp-clean-rewrite` 新开分支。

## Root handoff / 根目录回传信号

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-06-03 | status | QRP clean rewrite started on `feature/qrp-clean-rewrite`：源码拓扑迁移到 `0.1.0`，API/CLI 公共接口变为 `/api/0.1/...`，私有数据边界变为 `data/raw`、`data/interim`、`data/processed`、`reports/private_real`；旧 endpoint/command compatibility 不再是目标合同事实。 | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-04 | status | QRP clean rewrite 完成结构性本地阶段；private generated artifacts 仍为本地证据。 | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-15 | governance | Agent governance redesign 规定 `.agents/` 为项目治理根，global `.codex/` 为只读证据源，repo-local `.codex`/hooks/MCP/plugins 默认为 draft-only。 | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-15 | governance | 第二轮治理修补要求人读文件中文化、内存预算以代码 `0.80` 为事实源、短提示词精准路由、token economy 和外部工具共存规则。 | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-15 | governance | `feature/agent-governance-redesign` 已 fast-forward merge 到 `feature/qrp-clean-rewrite`；clean baseline 后续进入本地治理验证和 clean -> main gate 决策。 | WORKSPACE_STATUS, PLAN_INDEX | open |
