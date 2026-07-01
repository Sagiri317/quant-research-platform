# Verification and Gates

本文件定义治理验证、状态语义和 gate 分层。

## 1. 状态语义

- `PASS`：实际运行并通过，必须有 command、scope、exit result 或 artifact。
- `FAIL`：实际运行并失败，必须有失败范围和修复建议。
- `NOT_RUN`：未运行。
- `BLOCKED`：环境、依赖、权限、超时或边界阻塞。

任何报告不得把 `NOT_RUN` 或 `BLOCKED` 写成 `PASS`。

## 2. PASS 不可替代

以下结果互不替代：

- pytest PASS
- governance script PASS
- code-map flow PASS
- handoff verifier PASS
- single handoff verifier safety scan PASS
- source-tree scan PASS
- compare PASS
- sidecar PASS

每个 PASS 只覆盖对应命令和对应 scope。

## 3. 必要 gate

### Diff gate

```bash
git diff --check
```

检查 patch 基础格式，不代表功能正确。

### Governance gate

```bash
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
```

这是官方 active-doc state gate。clean tree 也必须扫描非空 active docs；`checked_files=[]` 不得 `PASS`。

Diff-only governance gate 使用：

```bash
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope changed-files --base-ref <base> --json
```

`--allow-active-docs` 和 `--scan-active-docs` 不是当前有效参数。

内部治理规则事实源是：

```text
.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py
```

active doc roots、current-contract-only 扫描规则、版本面规则、negative-test allowlist、source-tree scan allowlist、package safety path parts 和 status token 必须从这里读取或派生。

handoff required receipt paths、hard-scan replay rules、advisory review consistency 和 status token 也必须从 `governance_contract.py` 读取或派生，不得在 builder、verifier 或 tests 中复制平行规则表。

Hard-scan receipt gate 使用：

```bash
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --hard-scan-suite --receipt-out .run/<purpose>/receipts/hard-scan-receipt.json
```

该模式只写 compact receipt，不写 raw grep output。verifier 必须从 `final-branch.bundle` 重放 hard scan 并比对 `scan_id`、`rule_set_id`、count 和 `result_sha256`。

### Code-map gate

```bash
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
```

该 gate 验证当前代码地图和长期合同。

### Agent and architecture tests

```bash
python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast
```

如果完整运行超时，必须写 `BLOCKED` 并记录超时，可以补充分段运行结果。

Handoff receipt flow 的低成本 gate 放在 `tests/agents/handoff/`：覆盖 task 大 evidence 拒绝、status-only/placeholder/raw output 拒绝、generic smoke PASS 拒绝、advisory PASS 缺 receipt 拒绝、package/private/check receipt schema，以及从 bundle 重放 hard scan。

## 4. Active-doc scan

active docs 的唯一可执行路径清单是 `governance_contract.ACTIVE_GOVERNANCE_DOC_ROOTS`，其中包含 handoff tooling README。其他脚本和测试只读取或验证该常量，不维护平行清单。

active docs 只描述当前规则、职责、路径和命名。

current-contract-only 边界回归由 gate 记录为 `FAIL`；完整路径规则只由 `governance_contract.py` 定义。

命名边界扫描表面包括：

```text
file/path names
Python function/class/fixture/constant/module-level assignment names
JSON/YAML/TOML key names
static string concatenation or simple join synthesis
active docs current-truth wording
```

版本面规则由 `governance_contract.py` 和对应 tests 控制。

QRP-owned JSON 必须是顶层 object，必须包含 `schema_id`，且 `schema_id` 不编码版本。

## 5. Current-Contract Surface

current-contract-only 边界由 owner-layer negative tests 或明确 allowlist 控制。allowlist 必须说明目的和移除条件。

内部接口 gate 检查：

- 跨文件 private symbol import。
- `__all__` 单下划线导出。
- owner-local internal API 从 package `__init__.py` 重导出。
- storage/artifacts private module pattern。
- tests 白盒调用 private helper 分类为 `test_white_box_allowed`。

parallel/stale implementation gate 扫描 legacy、compat、deprecated、migration、old、new、v2、shim、fallback、wrapper 和 alias。真问题必须合并或删除；valuable deferred feature 必须有 owner、rationale、lifecycle 和 reference。

测试位置 gate：

- architecture boundary tests 放 `tests/architecture/`。
- handoff builder/verifier/evidence tests 放 `tests/agents/handoff/`。
- governance CLI/规则行为 tests 放 `tests/agents/`。
- A 股 processed owner behavior tests 放 `tests/markets/a_share/`。
- research behavior tests 放 `tests/research/`。
- artifact/storage/compute/workflow behavior tests 放对应 owner 目录。

A 股 owner-local naming gate 禁止 `markets/implementations/a_share/**` 内部 symbol、常量、文件名和 error code 重复 `a_share` / `ashare` / `china_a` / `china_share` 前缀；字符串 id 需要分类，不机械删除。

关键测试入口：

```text
tests/agents/test_verify_governance.py
tests/architecture/test_governance_contract_rg_gate.py
tests/architecture/test_share_safe_evidence.py
tests/architecture/test_share_safe_evidence_data_contract.py
tests/architecture/test_platform_topology.py
tests/contracts/test_platform_contracts.py
```

handoff 测试分层入口：

```bash
python tests/tools/run_profile.py --list-profiles
python tests/tools/run_profile.py --profile <profile> --run-dir .run/<purpose>/<profile>
```

## 6. Final report schema

最终报告至少包含：

```text
baseline
changed files
deleted files
commands run
status table
validation artifacts
remaining issues
working tree status
```
