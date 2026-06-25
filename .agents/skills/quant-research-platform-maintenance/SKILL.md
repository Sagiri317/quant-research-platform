---
name: quant-research-platform-maintenance
description: QRP 仓库维护 skill。用于 governance、handoff、contracts、docs、tests、agent workflow、REVIEW_SAFE evidence、architecture gates 和 review-fix loop。不要用于业务策略扩张、生产交易、broker/exchange 连接或 private data 读取。
---

# QRP Maintenance Skill

## 0. 触发范围

使用本 skill 的任务包括：

- governance docs、agent instructions、prompt templates、handoff workflow
- `verify_governance.py`、`verify_code_map_flow.py`、handoff builder/verifier
- `tests/agents/**`、`tests/architecture/**`、retired-surface negative tests
- docs hard cutover、open issues、REVIEW_SAFE evidence
- current-contract-only、private data safety、PASS taxonomy

不要用本 skill 直接处理 factor/model/backtest/storage runtime 的业务正确性，除非用户明确授权并指定 scope。

## 1. 必做 Preflight

任何操作前运行：

```bash
git status --short --branch
git log -1 --date=iso-strict --format="%H%n%cd%n%s"
git branch -a
```

当前事实只来自本轮用户指令、live git baseline 和当前仓库文件。

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

## 3. 当前分层事实

`policy/` 管共享安全、ID 安全和状态聚合。`contracts/` 只管结构合同。`catalog/` 只管 generic registry class。`profiles/` 是 API、CLI、agent、workflow 共用组合根。`research/` 只构建 result，不写文件。`reports/` 只 build/render evidence/report 内容。`artifacts/` 只写 LOCAL_ONLY run outputs 和 REVIEW_SAFE evidence 文件。`workflows/` 只编排 research -> reports -> artifacts。`storage/` 只管 dataset IO 和 data layout。

`.agents` 是 tracked tooling source。`.run` 是 output/evidence layer。禁止把 `.run` 作为官方工具入口。

## 4. 验证纪律

所有报告只用：`PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。

`PASS` 必须绑定实际命令或 artifact。完整测试超时写 `BLOCKED` 并说明超时，不能写 PASS。分段测试通过只能覆盖对应分段。顶层 PASS 不能掩盖子项 NOT_RUN 或 BLOCKED。

内部治理规则事实源是：

```text
.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py
```

`verify_governance.py`、handoff source-tree scan、handoff verifier 和相关 tests 必须引用该事实源或由它派生规则。retired governance paths 存在即 `FAIL`。retired public surface、retired naming、path/name/symbol/structured-key/synthesized-string patterns、version surface policy、stale truth、negative-test allowlist、source-tree scan allowlist 和 package forbidden path parts 不得在 builder、verifier 或 tests 中复制平行表。

建议命令：

```bash
git diff --check
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python -m pytest tests/agents tests/architecture -q
```

需要区分 fast 与依赖慢集成测试时，使用：

```bash
python -m pytest tests/architecture/test_share_safe_evidence.py -q
python -m pytest tests/architecture/test_share_safe_evidence_data_contract.py -q -m data_dependency
python -m pytest tests/agents/handoff tests/architecture/test_handoff_package_contract.py -q -m "not slow_integration"
python -m pytest tests/agents/handoff/test_build_handoff_package_integration.py -q -m slow_integration
```

## 5. Handoff 纪律

handoff scripts 位于：

```text
.agents/skills/quant-research-platform-maintenance/scripts/handoff/
```

`verify_handoff_package.py` 是唯一 handoff/package verifier。builder 只 build 并调用 verifier，不定义 sanitizer、forbidden member、private regex 或状态聚合规则。handoff artifact 输出到 `.run/` 或用户指定输出目录，不得把 `.run/` 中的历史产物当当前合同。

`.agents/` 只承载 governance、handoff 和 prompt tooling；runtime 业务逻辑必须留在 `src/quant_research_platform/` 的 owner layer。

## 6. 修改边界

修改 docs 时，如果删除或改名 active docs，必须同步修治理脚本和测试。不要为了让测试继续通过而恢复过期事实源。

修脚本时，优先修 state-based scan 和测试覆盖。业务逻辑文件默认禁止修改，除非用户明确授权或当前任务本身要求迁移业务层边界。

## 7. 结束报告

最终报告必须包含：

- branch 和 HEAD
- changed files
- deleted files
- commands run
- 每条命令的 PASS/FAIL/NOT_RUN/BLOCKED
- remaining issues
- working tree status
