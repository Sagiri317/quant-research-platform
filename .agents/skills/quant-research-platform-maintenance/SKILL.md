---
name: quant-research-platform-maintenance
description: QRP 仓库维护 skill。用于 governance、handoff、contracts、docs、tests、agent workflow、REVIEW_SAFE evidence、architecture gates 和 review-fix loop。不要用于业务策略扩张、生产交易、broker/exchange 连接或 private data 读取。
---

# QRP Maintenance Skill

## 0. 触发范围

使用本 skill 的任务包括：

- governance docs、agent instructions、prompt templates、handoff workflow
- `verify_governance.py`、`verify_code_map_flow.py`、handoff builder/verifier
- `tests/agents/**`、`tests/architecture/**`、current-contract negative tests
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

只读相关文件，不要为了“全面”把所有 docs、archived reports 或 `.run/` 全量读入上下文。

## 3. 当前分层事实

`policy/` 管共享安全、ID 安全和状态聚合。`contracts/` 只管结构合同。`catalog/` 只管 generic registry class。`profiles/` 是 API、CLI、agent、workflow 共用组合根。`research/` 只构建 result，不写文件。`reports/` 只 build/render evidence/report 内容。`artifacts/` 只写 LOCAL_ONLY outputs 和 REVIEW_SAFE evidence 文件。`workflows/` 只编排 research -> reports -> artifacts。`storage/` 只管 dataset IO 和 data layout。

运行结果合同是 `<run_root>/run.json`、`data/`、`outputs/<output_id>/`、`logs/` 和 `summary.md`。QRP-owned JSON 必须是 object，并使用不带版本后缀的 `schema_id`。compare 默认按 `output_id` 配对。状态聚合只由 `policy/check_status.py` 定义；`governance_contract.py` 只拥有扫描规则。

`.agents` 是 tracked tooling source。`.run` 是 output/evidence layer；官方工具入口留在 tracked source 中。

## 4. 验证纪律

所有报告只用：`PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。

`PASS` 必须绑定实际命令或 artifact。完整测试超时写 `BLOCKED` 并说明超时，不能写 PASS。分段测试通过只能覆盖对应分段。顶层 PASS 不能掩盖子项 NOT_RUN 或 BLOCKED。

内部治理规则事实源是：

```text
.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py
```

`verify_governance.py`、source-tree scan 和相关 tests 必须引用该事实源或由它派生规则。current-contract-only 规则、版本面规则、negative-test allowlist、source-tree scan allowlist、package safety path parts 和 status token 只从该事实源读取或派生。

建议命令：

```bash
git diff --check
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast
```

需要区分 fast、contract、architecture、integration 和 mutation 时，使用
`tests/tools/run_profile.py` 的 profile，不要在 skill 中复制 pytest marker
expression。fast/contract 用于日常收口；integration 和 mutation 必须显式选择，
并由对应 profile 承载。负向测试放在 owner layer：handoff 与 validation
evidence 放 `tests/agents/handoff/`，governance current-contract surface 放
`tests/agents/` 或 `tests/architecture/`，import boundary 放 `tests/architecture/`。

## 5. Handoff 纪律

handoff scripts 位于：

```text
.agents/skills/quant-research-platform-maintenance/scripts/handoff/
```

`verify_handoff_package.py` 是唯一 handoff/package verifier。builder 只 build 并调用 verifier，不定义 sanitizer、package member safety、private regex 或状态聚合规则。handoff artifact 输出到 `.run/` 或用户指定输出目录，当前合同来自本轮实际生成和验证的 artifact。

handoff package layout、sidecar、task metadata 和 validation summary 合同由 `handoff/handoff_contract.py` 拥有。receipt schema、receipt invariant 和 receipt 聚合由 `handoff/receipt_contract.py` 拥有。public machine replay 由 `handoff/replay_evidence.py` 执行，但 replay 模块不定义合同或 compare 规则。`governance_contract.py` 只拥有治理扫描、source-tree scan、共享安全路径和 status token 规则。

`.agents/` 只承载 governance、handoff 和 prompt tooling；runtime 业务逻辑必须留在 `src/quant_research_platform/` 的 owner layer。

## 6. 修改边界

修改 docs 时，如果删除或改名 active docs，必须同步修治理脚本和测试。测试修复不能通过恢复被替换事实源完成。

修脚本时，优先修 state-based scan 和测试覆盖。业务逻辑文件默认留在任务范围外，除非用户明确授权或当前任务本身要求迁移业务层边界。

## 7. 结束报告

最终报告必须包含：

- branch 和 HEAD
- changed files
- deleted files
- commands run
- 每条命令的 PASS/FAIL/NOT_RUN/BLOCKED
- remaining issues
- working tree status
