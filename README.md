# Quant Research Platform（QRP）

QRP 是一个面向多市场扩展的本地量化研究平台。当前启用市场是 A-share（A 股），当前阶段是 local research platform（本地研究平台），不是 paper trading（模拟交易）、live trading（实盘交易）或生产交易系统。

## 当前目标

当前 clean rewrite 的目标是建立一条可验证、可扩展、private-safe（私有数据安全）的研究链路：

```text
market/source data
-> processed dataset
-> factor / label / filter / context
-> research dataset
-> evaluation / diagnostics
-> model / prediction
-> report artifacts
-> public-safe evidence / handoff
```

当前 public surface（公共接口面）保持最小：

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

Approved private local CLI（获批私有本地命令行）只有：

```text
qrp-private schema-inventory --project-root . --output <path>
qrp-private evidence run
```

`qrp-private evidence run` 是 private local verification harness，用于本地
smoke/evidence generation；不是 public CLI；不是 public execution mode；不是
public evaluate、predict 或 report command；不是 paper/live trading；不承诺收益；
不输出 row-level private data。

## 治理入口

AI agents 必须先读根目录 `AGENTS.md`。项目内治理根是 `.agents/`，不是 repo-local `.codex/`。

常用入口：

```text
AGENTS.md
.agents/briefs/current-qrp-status.md
.agents/briefs/repo-map.md
.agents/briefs/user-principles.md
.agents/skills/quant-research-platform-maintenance/SKILL.md
.agents/skills/quant-research-platform-maintenance/references/task-modes.md
.agents/skills/quant-research-platform-maintenance/references/token-economy.md
.agents/skills/quant-research-platform-maintenance/references/templates/README.md
```

日常任务不要默认读取所有 docs、templates、`.run` evidence 或旧 handoff。先做 task mode routing（任务模式路由），再按需加载最小上下文。

## 工程原则

- Semantic closure（语义收口）：接口、命名、contracts、tests、docs、artifacts、handoff 证据必须一致。
- Naming first-class（命名一等）：命名是架构问题。
- Replace over parallel（替代优先）：替代旧结构时删除旧路径，不保留 legacy alias / compatibility shim。
- Minimal public API（最小公共接口）：不默认新增 API、CLI、registry、store、facade。
- Evidence over claims（证据优先）：未运行就是 `NOT_RUN`，失败就是 `FAIL`。
- Private-data safety（私有数据安全）：public-safe evidence 不包含行级私有数据。
- Layer-local change（局部改动）：新功能主要改 owner layer，跨层大改先做架构收口。

## Python 与资源策略

项目 Python 要求以 `pyproject.toml` 为准。当前 clean baseline 使用 `>=3.13,<3.14`；用户本地 Python 为 `3.13.14`。

运行时内存预算以代码事实源为准：`src/quant_research_platform/runtime/resources.py` 中 `DEFAULT_MEMORY_BUDGET_FRACTION = 0.80`，并可通过 `QRP_MEMORY_BUDGET_FRACTION` 覆盖。不要在治理文档中维护第二套不同数字。

## 外部工具边界

用户本地可能已有 Context7 MCP、GitHub MCP、Binance plugin、superpowers skills 和其他 global skills。QRP repo 不修改这些全局能力，也不创建 repo-local `.codex/`。外部工具可以辅助查询和审查，但不能绕过 QRP 的 task mode、owner layer、public surface、private safety 和 artifact truth gates。

## 私有数据边界

以下路径只用于本地私有数据或私有报告，不进入 public-safe bundle：

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
data/research/a_share/
reports/private_real/
```

review / handoff 包不得包含 parquet、csv、pkl、pickle、feather、arrow、raw vendor data、row-level private data、credentials、tokens 或本地绝对路径。

## 进一步阅读

- `PROJECT_STATUS.md`：当前阶段和下一步。
- `docs/architecture/agent-governance.md`：智能体治理架构。
- `docs/architecture/agent-governance-file-guide.md`：用人话解释治理目录和文件。
- `docs/architecture/open-issues-register.md`：开放问题登记。
