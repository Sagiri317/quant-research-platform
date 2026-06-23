---
name: quant-research-platform-maintenance
description: QRP 仓库维护 skill。用于 governance、handoff、contracts、docs、tests、agent workflow、public-safe evidence、architecture gates 和 review-fix loop。不要用于业务策略扩张、生产交易、broker/exchange 连接或 private data 读取。
---

# QRP Maintenance Skill

## 0. 触发范围

使用本 skill 的任务包括：

- governance docs、agent instructions、prompt templates、handoff workflow
- `verify_governance.py`、`verify_code_map_flow.py`、handoff builder/verifier/safety scripts
- `tests/agents/**`、`tests/architecture/**`、retired-surface negative tests
- docs hard cutover、open issues、public-safe evidence
- current-contract-only、private safety、PASS taxonomy

不要用本 skill 直接处理 factor/model/backtest/storage runtime 的业务正确性，除非用户明确授权并指定 scope。

## 1. 必做 preflight

任何操作前运行：

```bash
git status --short --branch
git log -1 --date=iso-strict --format="%H%n%cd%n%s"
git branch -a
```

不得使用历史分支名、旧 commit、旧 PR 状态或旧 handoff 作为当前事实。

## 2. 读取顺序

先读：

```text
AGENTS.md
.agents/briefs/repo-map.md
.agents/briefs/user-principles.md
```

再按任务读 references：

```text
references/current-contract.md
references/task-routing.md
references/verification-and-gates.md
references/handoff-and-evidence.md
references/tooling-and-portability.md
references/prompt-patterns.md
```

只读相关文件，不要为了“全面”把所有 docs、old reports 或 `.run/` 全量读入上下文。

## 3. 事实源纪律

仓库不再保存单独的 status 文档或状态简报。当前状态来自 live git baseline、当前用户指令、当前 open issues 和可执行合同。

`.agents` 是 tracked tooling source。`.run` 是 output/evidence layer。禁止把 `.run` 作为官方工具入口。

## 4. 验证纪律

所有报告只用：`PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。

`PASS` 必须绑定实际命令或 artifact。完整测试超时不能写 PASS。分段测试通过只能覆盖对应分段。

建议命令：

```bash
git diff --check
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --allow-active-docs --json
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python -m pytest tests/agents tests/architecture -q
```

## 5. handoff 纪律

handoff scripts 位于：

```text
.agents/skills/quant-research-platform-maintenance/scripts/handoff/
```

builder、verifier、source-tree scan、package safety 是不同 gate，不能互相替代。handoff artifact 输出到 `.run/` 或用户指定输出目录，不得把 `.run/` 中的历史产物当当前合同。

## 6. 修改边界

修文档时，如果删除或改名 active docs，必须同步修治理脚本和测试。不要为了让测试继续通过而恢复过期事实源。

修脚本时，优先修 state-based scan 和测试覆盖。业务逻辑文件默认禁止修改。唯一例外是 reports/evidence layer 在 public-safe evidence 路径硬编码需要最小同步时，可在报告中单独说明。

## 7. 结束报告

最终报告必须包含：

- branch 和 HEAD
- changed files
- deleted files
- commands run
- 每条命令的 PASS/FAIL/NOT_RUN/BLOCKED
- remaining issues
- working tree status
