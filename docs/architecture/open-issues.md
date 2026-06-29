# Open Issues

本文只记录当前仍需处理或已确认 deferred 的问题。

## 读取规则

- `open`：当前需要处理，或会阻塞后续任务。
- `deferred`：已确认重要，但不阻塞当前任务。
- 已完成事项不写入本文。
- 本文不是当前分支、当前阶段、当前 PR 或当前 merge gate 的事实源。

## Issue 格式

```text
ID:
Severity: P0 / P1 / P2 / P3
Owner layer:
Status: open / deferred
Evidence:
Next action:
Acceptance gate:
Scope boundary:
```

## P0 Current Blockers

暂无。

## P1 Current Issues

暂无。

## P2 Deferred

ID: reports-storage-dataset-boundary
Severity: P2
Owner layer: reports / workflows
Status: deferred
Evidence: `src/quant_research_platform/reports/verification_summary.py` and `src/quant_research_platform/reports/schema_inventory.py` currently import `quant_research_platform.storage.datasets`.
Next action: move dataset inspection to workflow or diagnostics owner, then pass only prepared payloads into reports rendering.
Acceptance gate: enable reports -> storage.datasets import boundary in owner-layer architecture tests without changing public report schema or workflow semantics.
Scope boundary: do not change report schema or workflow semantics without explicit approval.

ID: ml-validation-and-trial-contract-owner
Severity: P2
Owner layer: research / architecture
Status: deferred
Evidence: current model helpers own supervised fitting metadata and deterministic estimator dispatch, but no validation split or trial selection contract owner exists yet.
Next action: when validation and trial selection primitives are introduced, add owner-layer tests for train/validation/test date ordering, asset-date disjointness, test-metric exclusion from best-trial selection, and seeded trial sequence determinism.
Acceptance gate: contract tests derive expected behavior from the selected owner primitives without adding a tuning framework in tests.
Scope boundary: do not introduce a tuning framework, DL dependency, public registry, or model registry as part of a test-speed guardrail goal.

## P3 Watchlist

暂无。
