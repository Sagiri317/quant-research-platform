# Handoff and Evidence

本文定义 handoff 和 evidence 的当前合同。顶层 docs 只写 owner 原则；本文件可以列出具体 handoff layout。

## 1. 工具和 owner

tracked handoff tooling 位于：

```text
.agents/skills/quant-research-platform-maintenance/scripts/handoff/
```

`.run/` 只是 artifact output 位置，不是合同源。

- `handoff_contract.py` 拥有 zip layout、sidecar shape、task metadata shape、validation-summary rows、allowed/required/forbidden entries、sidecar status summary 和 exact package member set。
- `receipt_contract.py` 拥有 package-smoke、hard-scan、check-receipts、advisory-review receipt schema、receipt invariant 和 receipt status aggregation。
- `verify_handoff_package.py` 是唯一 package-level trust boundary。
- `governance_contract.py` 只拥有治理扫描、source-tree scan、共享安全路径、status token 和 retired surface scanning rules。

## 2. Minimal Zip Layout

正式 handoff zip 只允许包含：

```text
artifact_manifest.json
final-branch.bundle
task_metadata.json
handoff_evidence/validation-summary.json
handoff_evidence/package-smoke-receipt.json
handoff_evidence/hard-scan-receipt.json
handoff_evidence/check-receipts.json
handoff_evidence/advisory-review-receipt.json
```

`handoff_evidence/advisory-review-receipt.json` 是 optional，只有 advisory review 实际运行时才存在。未运行时不生成空 advisory receipt。

## 3. Sidecar

sidecar JSON 只允许包含：

```text
schema_id
zip_filename
zip_sha256
artifact_manifest_sha256
branch
base
final_head
generated_at_utc
status_summary
advisory_reviews
```

`status_summary` 只允许包含：

```text
validation
scripted_checks
package_smoke
profile_smoke
private_validation
baseline_compare
governance
code_map
handoff_verify
share_safety_scan
validation_blocked_count
validation_fail_count
```

未运行 advisory review 时，`advisory_reviews` 使用 `{}`。sidecar PASS 不代表 pytest、business smoke、compare 或 private validation PASS；每个 status 只覆盖对应 evidence。

## 4. Validation Summary

`handoff_evidence/validation-summary.json` 的 `schema_id` 必须是 `qrp-handoff-validation-summary`。每行只允许：

```text
name
status
evidence
```

mandatory rows：

```text
scripted_checks
package_smoke
profile_smoke
private_validation
baseline_compare
governance
code_map
```

`scripted_checks` 只表示 `check-receipts.json` 聚合状态，不表示全量 tests。`private_validation` 不使用单独 status-only 文件；未运行或阻塞时，`evidence` 写明原因；实际 PASS/FAIL 时，`evidence` 引用 `handoff_evidence/check-receipts.json#private-validation`。

`validation-summary.json` is a derived view. Every `PASS`, `FAIL`, or `BLOCKED` row must be backed by a receipt graph edge in `check-receipts.json`, `package-smoke-receipt.json`, or `hard-scan-receipt.json`.
Mandatory receipt-backed rows may not use reason-only evidence; only optional command rows may be `NOT_RUN` with a `not_run:` reason when the command receipt is absent.

## 5. Receipt Invariants

receipt 必须是 compact REVIEW_SAFE JSON，不包含 stdout、stderr、raw output、raw grep output 或 grep hits。

`check-receipts.json` 包含 `qrp-check-receipt` rows。每行必须包含 `command_argv` list，`command` 只是由 argv 生成的人读展示字符串；`counts` key 必须存在，非 pytest 命令可为 `null`。`PASS` 必须 `exit_code == 0`，`FAIL` 必须 `exit_code != 0`。`PASS` 时 pytest counts 的 failed 和 errors 必须为 0；`collected=0` 但 outcome count 非零必须拒绝。`NOT_RUN` 必须有 not-run reason，`BLOCKED` 必须有 blocked reason。

`hard-scan-receipt.json` 只保存 scan id、rule set、count、result hash 和 classified exceptions；verifier 必须从 `final-branch.bundle` 重放 hard scan 并比对。

`package-smoke-receipt.json` 只保存 wheel/sdist 文件名与 hash、临时 no-deps install/import 状态、content scan 状态和 forbidden count；wheel、sdist、临时 install target 不进入 handoff。

## 6. Forbidden Entries

verifier 必须拒绝退役 handoff entry，包括：

```text
git_status.txt
handoff_evidence/generation.json
handoff_evidence/bundle-reconstruction.json
handoff_evidence/private-smoke-status.json
handoff_evidence/private-validation-status.json
receipt-index.json
```

旧 entry 名称只能出现在 `handoff_contract.py` 或 `governance_contract.py` 的 retired/forbidden rule tuple，以及对应 negative tests。

## 7. Bundle and Manifest

`artifact_manifest.json` 位于 zip 根部，记录除自身外所有 zip members 的 path、size 和 sha256；entries 必须与实际 zip members 精确一致。sidecar 的 `artifact_manifest_sha256` 和 `zip_sha256` 必须匹配原始字节 hash。
`artifact_manifest.json` is exact-schema and only contains `schema_id` plus `entries`; verifier rejects tool source claim fields.

`final-branch.bundle` 必须可 `git bundle verify`，可 clone 指定 branch，clone 后 HEAD 必须等于 sidecar `final_head`，working tree 必须 clean。verifier 直接复验 bundle，不依赖额外 reconstruction evidence。

## 8. Mutation Tests

full package mutation tests 位于 `tests/agents/handoff/test_handoff_package_mutations.py`。Mutation 后必须刷新 `artifact_manifest.json`、sidecar `artifact_manifest_sha256` 和 sidecar `zip_sha256`，证明 verifier 是因 contract violation 失败，不是 hash mismatch 失败。
