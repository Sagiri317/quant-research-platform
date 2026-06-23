# Open Issues

本文件是当前 open 和 deferred 问题登记。它不保存 done、resolved、historical notes 或路线图。历史追溯使用 git history。

## 读取规则

- `open`：当前需要处理或会阻塞后续任务。
- `deferred`：已确认重要，但不阻塞当前任务。
- 不记录已完成问题。
- 不把本文当当前分支或当前阶段事实源。

## Issue 格式

```text
ID:
Severity: P0 / P1 / P2 / P3
Owner layer:
Status: open / deferred
Evidence:
Next action:
Acceptance gate:
Forbidden scope:
```

## P0 Current Blockers

暂无。若出现会导致错误执行、错误验证、私有数据泄漏或旧合同复活的问题，应在这里登记，并在修复后从本文删除。

## P1 Deferred

### QRP-GOV-001

```text
Severity: P1
Owner layer: governance scripts
Status: deferred
Evidence: verify_governance 当前应支持 state-based active-doc scan，避免 clean tree 下 checked_files 为空仍 PASS。
Next action: Codex 在真实仓库中修改 verify_governance.py 并补 tests/agents/test_verify_governance.py。
Acceptance gate: active-doc scan 在 clean tree 上扫描非空 active docs；故意加入 stale active doc 时 FAIL。
Forbidden scope: 不改业务逻辑，不恢复状态文档。
```

### QRP-GOV-002

```text
Severity: P1
Owner layer: architecture tests
Status: deferred
Evidence: 文档硬切换删除历史 docs/reports 后，相关治理测试需要从保护历史文件改为保护 current contract。
Next action: Codex 同步修 tests/agents/**、tests/architecture/** 和必要 reports evidence tests。
Acceptance gate: targeted governance tests PASS；完整测试若环境超时必须写 BLOCKED/TIMEOUT。
Forbidden scope: 不改 factor/model/backtest/storage runtime。
```

## P2 Watchlist

### QRP-GOV-003

```text
Severity: P2
Owner layer: handoff tooling
Status: deferred
Evidence: handoff source-tree scan 可进一步增加 stale active-doc patterns。
Next action: 后续在 handoff builder/verifier 任务中处理。
Acceptance gate: source-tree scan 对旧状态事实、旧 tool source 和 retired fields 有独立 scan group。
Forbidden scope: 不改变 handoff artifact 公共合同，除非同步 schema 和 tests。
```
