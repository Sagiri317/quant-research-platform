# QRP

QRP 是一个本地量化研究平台，用于组织数据合同、因子研究、特征处理、模型训练、预测、评估、组合构建、报告与 evidence。它面向研究和演示，不是生产交易系统，不接 broker 或 exchange，不提供 investment advice。

## 当前定位

QRP 的长期定位是：用严格合同、可复现 evidence 和受控 agent workflow 支撑量化研究迭代。仓库文档只描述长期合同和操作规则，不记录当前分支、当前阶段、当前 PR、最近运行结果或下一步任务。

当前状态必须从 git 和当前任务现场读取：

```bash
git status --short --branch
git log -1 --date=iso-strict --format="%H%n%cd%n%s"
```

## 目录概览

```text
.agents/     agent 指令、skills、governance scripts、handoff tooling
src/         QRP 源码
tests/       当前合同测试、治理测试、架构测试
docs/        当前合同文档和代码地图
data/        本地数据占位说明，不提交私有数据
.run/        本地运行输出和 evidence，不作为 tracked tooling
```

## 当前 public surface

当前 public API 只有：

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

当前 public CLI 只有：

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

## agent 入口

coding agent 应先读 `AGENTS.md`，再按任务范围读取 `.agents/skills/quant-research-platform-maintenance/SKILL.md`。需要深入执行时，再按需读取 `.agents/skills/quant-research-platform-maintenance/references/**`。

`AGENTS.md` 不是状态板。仓库不再保存单独的状态文档或状态简报。需要当前事实时，必须使用 live git baseline。

## 文档原则

- 文档保存当前合同，不保存历史叙事。
- open issues 只保存当前 open 或 deferred 问题。
- 历史 evidence 不进入 `docs/` 作为长期事实源。
- `.agents` 是 tracked tooling source。
- `.run` 是 output layer 和 evidence layer。

## 验证原则

所有验证结果必须使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。`PASS` 必须对应实际命令或可检查 artifact。未运行的检查不能写成 PASS。

常用治理验证：

```bash
git diff --check
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --allow-active-docs --json
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python -m pytest tests/agents tests/architecture -q
```

## 数据安全

禁止提交 private data、vendor marker、本地绝对路径、股票代码样本、row-level sample 或 raw traceback。公开 evidence 只能包含 public-safe 摘要、schema、计数、hash、命令状态和路径结构。
