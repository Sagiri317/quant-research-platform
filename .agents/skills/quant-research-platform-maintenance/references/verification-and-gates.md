# Verification and Gates

本文件定义治理验证、状态语义和 gate 分层。

## 1. 状态语义

- `PASS`：实际运行并通过，必须有 command、scope、exit result 或 artifact。
- `FAIL`：实际运行并失败，必须有失败范围和修复建议。
- `NOT_RUN`：未运行。
- `BLOCKED`：环境、依赖、权限、超时或边界阻塞。
- `TIMEOUT`：命令实际超时。

任何报告不得把 `NOT_RUN`、`BLOCKED` 或 `TIMEOUT` 写成 `PASS`。

## 2. PASS 不可替代

以下结果互不替代：

- pytest PASS
- governance script PASS
- code-map flow PASS
- handoff verifier PASS
- package safety PASS
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

active doc roots、retired governance paths、retired naming、stale truth、retired public surface、negative-test allowlist、source-tree scan allowlist、package forbidden path parts 和 status token 必须从这里读取或派生。

### Code-map gate

```bash
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
```

该 gate 应验证当前代码地图和长期合同，不应要求历史阶段锚点。

### Agent and architecture tests

```bash
python -m pytest tests/agents tests/architecture -q
```

如果完整运行超时，必须写 `BLOCKED/TIMEOUT`，可以补充分段运行结果。

## 4. Active-doc scan

active docs 至少包括：

```text
AGENTS.md
README.md
.agents/briefs/**
.agents/skills/quant-research-platform-maintenance/SKILL.md
.agents/skills/quant-research-platform-maintenance/references/**
docs/architecture/**
docs/code-map/**
docs/api/**
docs/data/**
data/README.md
```

active docs 不得把历史 branch、历史阶段、历史 PR、历史 reports 当当前事实。

Retired governance paths 存在即 `FAIL`。至少包括：

```text
PROJECT_STATUS.md
.agents/briefs/current-qrp-status.md
docs/architecture/open-issues-register.md
docs/reports/clean-rewrite-0.1-*
```

## 5. Retired surface

旧字段、旧路径、旧 artifact key、旧 execution mode 只允许在 retired-surface negative tests 或明确 allowlist 中出现。allowlist 必须说明目的和移除条件。

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
