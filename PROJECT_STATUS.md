# QRP 项目状态

本文件记录当前开发状态，不替代代码、tests、contracts 或 `AGENTS.md`。AI agents 执行任务前仍必须用 `git status --short --branch` 和当前分支 diff 校验事实。

## 当前分支与阶段

| 项目 | 当前事实 |
|---|---|
| 长期主分支 | `main` |
| 当前 clean baseline | `feature/qrp-clean-rewrite` |
| 当前治理修补分支 | `feature/agent-governance-redesign` |
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
- 当前治理分支的重点是中文化、事实修正、token economy（token 节约）、短提示词精准路由和一遍过机制。

## 当前 blocker / 风险

1. Public surface（公共接口面）必须保持最小，除非有单独批准。
2. 私有数据和私有报告不得进入 public-safe artifact 或 review/handoff 包。
3. 未运行检查必须标记为 `NOT_RUN` 或 `BLOCKED`，不能写成 `PASS`。
4. 治理脚本需要在用户 Win11 + Python 3.13.14 环境最终验证；本仓库内 Linux/macOS/WSL 验证不能替代用户本机验证。
5. 当前分支若继续修改，应只追加治理层修补 commit，不要从 `feature/qrp-clean-rewrite` 重新生成一整套 patch。

## 下一步

1. 在 `feature/agent-governance-redesign` 上完成第二轮治理修补：中文化、事实修正、模板一致性、token economy、外部工具共存规则。
2. 用治理脚本、`py_compile`、`git diff --check` 和 targeted architecture/governance tests 验证。
3. 生成 follow-up patch / package 供 apply-only review。
4. 通过独立 review 后，再执行 merge gate；不要直接 merge 到 `main`。

## Root handoff / 根目录回传信号

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-06-03 | status | QRP clean rewrite started on `feature/qrp-clean-rewrite`：源码拓扑迁移到 `0.1.0`，API/CLI 公共接口变为 `/api/0.1/...`，私有数据边界变为 `data/raw`、`data/interim`、`data/processed`、`reports/private_real`；旧 endpoint/command compatibility 不再是目标合同事实。 | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-04 | status | QRP clean rewrite 完成结构性本地阶段；private generated artifacts 仍为本地证据。 | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-15 | governance | Agent governance redesign 规定 `.agents/` 为项目治理根，global `.codex/` 为只读证据源，repo-local `.codex`/hooks/MCP/plugins 默认为 draft-only。 | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-15 | governance | 第二轮治理修补要求人读文件中文化、内存预算以代码 `0.80` 为事实源、短提示词精准路由、token economy 和外部工具共存规则。 | WORKSPACE_STATUS, PLAN_INDEX | open |
