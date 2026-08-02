# Verification and Gates

本文定义治理验证、状态语义和 gate 分层。它只保存长期规则，不记录当前分支、当前 PR、最近运行结果或一次性 evidence。

## 1. 状态语义

- `PASS`：实际运行并通过，必须有 command、scope、exit result 或 artifact。
- `FAIL`：实际运行并失败，必须说明失败范围和修复建议。
- `NOT_RUN`：未运行。
- `BLOCKED`：环境、依赖、权限、超时或边界阻塞。

任何报告都不能把 `NOT_RUN` 或 `BLOCKED` 写成 `PASS`。

## 2. PASS 不可替代

以下结果互不替代：

- pytest PASS
- governance script PASS
- handoff verifier PASS
- share-safety PASS
- source-tree scan PASS
- compare PASS
- sidecar PASS

每个 `PASS` 只覆盖对应命令和对应 scope。

## 3. 必要 Gate

### Diff Gate

```bash
git diff --check
```

该 gate 只检查 patch 基础格式，不代表功能正确。累计分支检查需要明确使用 `base..HEAD`，不能用干净工作区的空 diff 代替。

### Governance Gate

```bash
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
```

这是 official active-doc state gate。即使工作区干净，也必须扫描非空 active docs；`checked_files=[]` 不能 `PASS`。

Diff-only governance gate 使用：

```bash
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope changed-files --base-ref <base> --json
```

内部治理规则事实源是：

```text
.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py
```

active doc roots、sensitive-text 精确例外、layer/writer owner 规则和 package safety path parts 必须从这里读取或派生。

source-tree sensitive scan 和 bundle share-safety scan 统一复用该事实源。source-tree scan 由 `SourceTreeScanInput` 驱动，只扫描明确的 tracked text roots；精确例外按 path、rule 和完整行匹配。

handoff layout gate 由 `handoff/handoff_contract.py` 和 `handoff/verify_handoff_package.py` 拥有。`FieldSpec` 和 exact schema validation 只由 `handoff_contract.py` 定义；`receipt_contract.py` 只拥有 receipt schema specs、invariants、normalization 和 comparison。source-tree sensitive boundary 由 `governance_contract.py`、`verify_governance.py` 和最小行为测试共同保护。builder、collector、verifier 和 tests 不得复制平行 receipt validator 或 handoff layout 表。

Hard-scan gate 由正式 handoff builder/verifier 闭合：

```bash
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py --repo-root . --out-dir .run/handoffs --task-name <task-name> --base-ref <base-ref>
```

builder 从 clean Git state 写入 bundle，从 bundle clone 自动生成 machine receipts 和 validation summary，再执行 single `HandoffPackagePayload` load 做 package payload validation，检查 schema、hash 和 share-safety。`verify_handoff_package_result()` 是唯一正式 package-level verifier result path。standalone verifier CLI 保留为完整 bundle replay 和 receipt comparison 的直接入口，但不是额外 Python wrapper。builder 和 standalone verifier 共用 `load_handoff_package_payload()` 与 `validate_handoff_package_payload()`；standalone verifier 在同一个 bundle clone 中复用 source-tree scan 结果执行 share-safety 和 hard-scan receipt replay。具体字段、结构、receipt replay、share-safety 和 hard-scan receipt construction 以 executable contracts、`verify_handoff_package.py` 和 `replay_evidence.py` 为准。`verify_governance.py` 只运行 governance checks，不写 handoff receipts。

package smoke receipt 使用 conda-managed non-isolated build：`python -m build --no-isolation --sdist --wheel --outdir <dist>`。构建后端依赖由 `environment.yml` 显式提供；构建 wheel 后，用标准库解压 wheel，并通过 `PYTHONPATH` 执行 import smoke。`profile_smoke` row 来自正式 profile runner 的 `profile-smoke` profile。

### Structure Audit Gate

```bash
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py --repo-root . --json
```

该 gate 只覆盖三项长期结构事实：`pyproject.toml` 与 `environment.yml` 的 dependency/Python/Ruff 同步、tracked `.agents` scripts 不依赖 tests，以及 handoff validation row/replay command ID 唯一且集合闭合。它不替代行为测试、package smoke 或 handoff verifier。

### Agent And Architecture Tests

```bash
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile all --run-dir .run/<purpose>/all -- --durations=100
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile all --run-dir .run/<purpose>/all-xdist -- -n 4 --dist loadfile --durations=100
```

`pytest-xdist` 只作为 dev-only 耗时诊断依赖，具体版本以 root config
当前声明为准。sequential `all` profile 是正式 truth gate；`all-xdist` 不能替代
`all`，只能补充耗时证据。

如果完整运行超时，必须写 `BLOCKED` 并记录超时；可以补充分段运行结果。

Handoff receipt flow 的低成本 gate 放在 `tests/agents/handoff/`：覆盖唯一 builder 入口、sidecar zip envelope、exact zip layout、status-only/placeholder/raw output 拒绝、package/check/hard-scan receipt schema、package mutation，以及从 bundle clone replay public machine receipts。mutation tests 对 payload/schema/replay mismatch 使用 `compare_replayed_handoff_payloads()` fast path；完整 bundle verifier 只保留给 valid package smoke、Git identity 和 sidecar envelope 变异。

Replay environment preflight 是 builder/verifier-local diagnostic gate。缺失依赖
或版本不满足时，builder 应 fail fast 且不写出 partial package；verifier 应在
bundle verify、clone、identity checks 和 source-tree share-safety scan 之后、receipt
replay 之前报告 `replay_environment_blocked:*`。该状态不进入 handoff zip、
receipt schema、validation row 或 artifact manifest。runtime preflight 的 Python
版本和 replay 必需 distribution 约束从 `pyproject.toml` 派生；`environment.yml`
只通过 structure audit / environment contract 检查与 `pyproject.toml` 同步。

Formal handoff validation gate 要求 mandatory validation rows 全部为 `PASS`。builder 和 verifier 都必须调用 `handoff_contract.mandatory_validation_failures()`；任一 mandatory row 为 `FAIL`、`NOT_RUN`、`BLOCKED`、缺失、重复或未知状态时，formal handoff 失败。`validation-summary.json` 顶层 status 由 rows 聚合派生，不得由 package layout PASS、receipt 自洽或 verifier status 替代 row-level PASS。

## 4. Active-Doc Scan

active docs 的唯一可执行路径清单是 `governance_contract.ACTIVE_GOVERNANCE_DOC_ROOTS`，其中包含 handoff tooling README。其他脚本和测试只读取或验证该常量，不维护平行清单。

active docs 只描述当前规则、职责、owner 和调用流。current-contract-only 边界回归由 owner 行为、真实调用流和 machine-consumed contract 裁决；词法或 AST 扫描只产生候选，不能仅凭名称、路径或语法返回 `FAIL`。

QRP-owned JSON 的顶层 identity 由对应 artifact contract 与行为测试控制：payload 必须是 object，并使用该 owner 当前定义的精确 `schema_id`。

## 5. Current-Contract Surface

current-contract-only 边界由 owner 行为测试、依赖方向、唯一 active registration 和公共 machine contract 控制。allowlist 必须说明真实例外和移除条件。平行实现按实际 caller、状态与副作用 owner 裁决；未接入的 deferred 代码和测试删除，只在仍有产品价值时保留 open issue。

测试通常随行为 owner 放置，便于定位和运行；目录、helper 可见性、`__all__` 形状或特定词汇本身不构成失败条件。

Hard gate 只保护依赖方向、唯一 active owner、公共 machine contract、资源生命周期、
REVIEW_SAFE 边界和 evidence truth；名称、文件位置与实现语法只产生审查候选。

handoff 测试分层入口：

```bash
conda run -n <environment.yml name> python tests/tools/run_profile.py --list-profiles
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile <profile> --run-dir .run/<purpose>/<profile>
```

## 6. Final Report Schema

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
