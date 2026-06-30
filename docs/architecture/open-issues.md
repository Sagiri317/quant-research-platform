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

ID: profile-runner-governance-temp-boundary
Severity: P1
Owner layer: tests / profile runner / governance tooling
Status: open
Evidence: `tests/tools/run_profile.py` allows user-selected `--output-dir` and `--tmp-dir`. When those directories are inside the repo, pytest-generated temporary files or negative-test samples can be scanned by `verify_governance.py` active-docs or state scans as if they were real repo files, causing false governance `FAIL` results.
Next action: add a controlled managed-roots mechanism where `run_profile.py` declares the current run's output and tmp roots to child processes, and `verify_governance.py` skips only those roots for that run after validating the roots are repo-contained, are not `src/`, `tests/`, `docs/`, `.agents/`, or other formal source directories, do not escape the repo, and do not hide tracked files.
Acceptance gate: governance scans do not fail on profile-owned temporary output roots, while real repo files and tracked files under formal source directories remain scanned.
Scope boundary: do not force all profile outputs into `.run/`; keep `--output-dir` and `--tmp-dir` user-selectable; do not permanently add `.review-profile/` to ignore lists. The post-merge handoff package having validation `NOT_RUN` is expected packaging evidence behavior, not a separate open issue or merge-gate validation result.

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

ID: handoff-integration-subprocess-cost
Severity: P2
Owner layer: tests / handoff tooling
Status: deferred
Evidence: `tests/agents/handoff/test_build_handoff_package_integration.py` still keeps real builder subprocess coverage for handoff package contract and custom evidence negative cases.
Next action: split lower-cost contract coverage from real package-build subprocess coverage only if the same malicious package and evidence rejection semantics remain covered.
Acceptance gate: handoff integration and mutation profiles keep package-build, verifier, and malicious-input coverage with fewer repeated subprocess/package-build operations.
Scope boundary: do not delete tests, skip tests, move tests out of their semantic profile, or change the public handoff schema to reduce runtime.

ID: windows-pytest-tmp-path-length
Severity: P3
Owner layer: tests / profile runner
Status: deferred
Evidence: Windows pytest runs can fail tests that create deeply nested data paths when an explicit `--tmp-dir` is long; the same tests pass with a shorter repo-relative tmp directory.
Next action: keep profile evidence tmp roots short on Windows, and if this recurs, add a runner or documentation guardrail that preserves explicit repo-relative tmp paths without shell wrappers or absolute-path assumptions.
Acceptance gate: full pytest profile passes on Windows with a path-length-safe tmp strategy while preserving all tests and profile coverage.
Scope boundary: do not add `.ps1` or shell wrappers, do not skip tests, do not change business `src/**`, and do not hardcode machine-specific paths.

## P3 Watchlist

暂无。
