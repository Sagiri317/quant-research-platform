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

ID: handoff-integration-subprocess-cost
Severity: P2
Owner layer: tests / handoff tooling
Status: deferred
Evidence: Source-tree scan input, active-doc repo state enumeration, handoff package payload loading, fixture naming, and mutation fast-path validation have been consolidated. Remaining cost is the intentionally retained real builder subprocess coverage in `tests/agents/handoff/test_build_handoff_package_integration.py`.
Next action: revisit only if after full-suite verification the real builder subprocess path is still a measured slowest-path blocker; do not split it unless the same package-build, verifier, Git identity, and malicious-input semantics stay covered.
Acceptance gate: full-suite evidence shows the retained real builder subprocess coverage is not a blocker, or an owner-layer split preserves equivalent handoff contract coverage with lower repeated subprocess/package-build cost.
Scope boundary: preserve handoff coverage, public handoff schema/layout, receipt fields, mandatory validation rows, and standalone verifier replay semantics.

## P3 Watchlist

ID: windows-pytest-tmp-path-length
Severity: P3
Owner layer: tests / profile runner
Status: deferred
Evidence: Windows pytest runs can fail tests that create deeply nested data paths when the profile directory is long; the same tests pass with a shorter repo-relative run directory.
Next action: keep profile directories short on Windows, and if this recurs, add a runner or documentation guardrail that preserves explicit repo-relative run paths without shell packaging scripts or absolute-path assumptions.
Acceptance gate: full pytest profile passes on Windows with a path-length-safe tmp strategy while preserving all tests and profile coverage.
Scope boundary: preserve Python-only test entrypoints, test coverage, business `src/**` semantics, and machine-independent paths.
