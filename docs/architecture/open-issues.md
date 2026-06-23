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

暂无。

## P2 Watchlist

### QRP-GOV-003

```text
Severity: P2
Owner layer: tests
Status: deferred
Evidence: 部分治理/架构测试收集路径仍可能受 data/parquet 依赖影响，独立审查环境缺少 `pyarrow` 时不应阻塞纯治理 fast tests。
Next action: 后续拆分纯治理 fast tests 与 data/parquet 依赖 tests。
Acceptance gate: 缺少 data/parquet 依赖时，纯治理 fast tests 仍可收集和运行；依赖型测试明确写 `BLOCKED` 或 marker。
Forbidden scope: 本议题不改业务数据读取语义，不移除必要的 data/parquet 合同测试。
```

### QRP-GOV-004

```text
Severity: P2
Owner layer: handoff tests
Status: deferred
Evidence: handoff integration test 较慢，完整运行未运行或超时时不能由分段 PASS 替代。
Next action: 后续区分 fast contract gate 与 slow integration test。
Acceptance gate: fast handoff contract gate 可快速运行；slow integration 单独标记，full integration 未运行/超时必须写 `NOT_RUN`、`BLOCKED` 或 `TIMEOUT`。
Forbidden scope: 不降低 builder、verifier、source-tree scan 或 package safety 的独立 gate。
```
