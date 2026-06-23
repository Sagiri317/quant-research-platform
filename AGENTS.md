# QRP Agent 指令

本文件是 QRP 仓库中面向 coding agent 的根指令。它只保存长期稳定的执行规则，不保存当前分支、当前阶段、最近运行结果、PR 状态或下一步任务。所有 agent 在计划、审查、修改、验证前，必须先从 live git baseline 重建现场事实。

## 1. 项目边界

QRP 是本地量化研究平台，目标是支持数据合同、因子研究、特征处理、模型训练、预测、评估、组合构建、报告和 evidence 管理。QRP 不是生产交易系统，不是 paper trading 系统，不是 live trading 系统，不接 broker，不接 exchange，不提供 investment advice。

未经用户明确批准，禁止新增 public API、CLI、registry、store、facade、scheduler、worker、deployment、external LLM API、vendor API、broker connector、exchange connector。未经用户明确批准，禁止访问 private data 或生成 row-level private-derived public output。

## 2. live git baseline

任何任务开始时，必须运行并记录：

```bash
git status --short --branch
git log -1 --date=iso-strict --format="%H%n%cd%n%s"
git branch -a
```

如果任务涉及远端分支或 merge，还必须检查 upstream、divergence 和目标分支。不得继承旧对话、旧 handoff、旧 sidecar、旧 PR body、旧 branch 名称或旧 commit 作为当前事实。

仓库文档不得声明当前分支、当前阶段、当前 PR 状态或当前 merge gate。需要当前事实时，使用 live git baseline 和当前用户指令。

## 3. 事实源优先级

执行优先级如下：

1. 用户本轮明确指令和 approved task envelope
2. live git baseline 与当前仓库文件
3. 本文件
4. `.agents/skills/quant-research-platform-maintenance/SKILL.md`
5. `.agents/briefs/repo-map.md` 和 `.agents/briefs/user-principles.md`
6. `.agents/skills/quant-research-platform-maintenance/references/**`
7. `docs/**` 的当前合同文档
8. 旧 handoff、旧 logs、旧 reports、旧对话摘要

如果文档和代码、tests、contracts、schemas、runtime behavior 冲突，以可执行合同和当前仓库状态为准；停止并报告冲突，不要用叙事文档覆盖现场事实。

## 4. `.agents` 与 `.run`

`.agents/skills/quant-research-platform-maintenance/scripts/` 是 tracked governance tooling。handoff builder、verifier、package safety、governance check 等工具必须从这里读取。

`.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 是内部治理规则事实源，集中定义 active governance docs、retired governance paths、retired naming、stale truth、package forbidden path parts 和 status token。其他治理 gate 可以分层执行，但不得复制一份平行规则表。

`.run/` 只是 output layer 和 evidence layer。它可以保存运行产物、handoff zip、manifest、logs、reports、temporary evidence，但不能反向定义架构、合同、工具入口或当前状态。

## 5. 验证状态语义

所有最终报告、sidecar、handoff、review、prompt 和 PR 总结只能使用以下状态：

- `PASS`：实际运行并通过，必须附 command、scope、result 或 artifact path。
- `FAIL`：实际运行并失败，必须说明失败范围和下一步。
- `NOT_RUN`：未运行，不得伪装为 PASS。
- `BLOCKED`：因环境、依赖、权限、超时或边界无法运行，必须说明阻塞条件。
- `TIMEOUT`：命令实际超时，不能写成 `PASS`。

pytest PASS、governance PASS、handoff verifier PASS、package safety PASS、source-tree scan PASS、compare PASS、sidecar PASS 不能互相替代。每个 PASS 只覆盖对应命令和对应 scope。

## 6. current-contract-only

QRP 采用 current-contract-only。普通 tests 只保护当前合同。旧字段、旧路径、旧 artifact 字段、旧执行模式、旧命名和兼容 alias 只能出现在 retired-surface negative tests 或明确 allowlist 中。除非用户批准兼容期，禁止 fallback read、silent drop、compatibility field 和 alias field。

新实现应替代旧实现，不应并行保留两个语义相同的入口。确需并行时，必须说明语义差异、生命周期、owner layer 和未来收敛方式。

Retired governance paths 不得复活。`PROJECT_STATUS.md`、`.agents/briefs/current-qrp-status.md`、`docs/architecture/open-issues-register.md` 和 `docs/reports/clean-rewrite-0.1-*` 只要在 tracked tree 或 working tree 中存在，治理 gate 必须 `FAIL`。

## 7. private safety

禁止把 private data、vendor marker、本地绝对路径、股票代码样本、raw traceback、row-level sample 写入 public-safe docs、handoff、reports、logs 或 final response。需要引用 private 验证时，只写摘要、计数、hash、schema、command status 和 public-safe evidence。

## 8. 修改边界

默认允许修改与任务直接相关的最小文件集合。修改 docs、governance scripts、handoff scripts、tests 时，必须同步更新相关验证。修改业务逻辑时，必须有用户明确授权。

禁止为了让测试通过而恢复过期文档事实。遇到脚本仍引用已删除的历史状态文档时，应修脚本和测试，而不是恢复旧文档。

## 9. 推荐验证命令

按任务范围选择最小充分命令，不要把未运行项写成 PASS：

```bash
git diff --check
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python -m pytest tests/agents tests/architecture -q
```

如果完整测试因环境或超时不可行，必须分段运行并明确写 `BLOCKED` 或 `NOT_RUN`。

## 10. 提交流程

commit 前必须运行 `git status --short --branch` 并检查 diff。禁止 force push、main push、未经批准的 rebase、amend、squash、merge main。最终报告必须列出 branch、HEAD、changed files、validation results、remaining issues 和 working tree status。
