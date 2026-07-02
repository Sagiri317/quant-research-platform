# Handoff and Evidence

本文定义 handoff 和 REVIEW_SAFE machine evidence 的当前长期合同。顶层 docs 只写 owner 原则；本文件可以列出具体 handoff layout。

## 1. 工具和 owner

tracked handoff tooling 位于：

```text
.agents/skills/quant-research-platform-maintenance/scripts/handoff/
```

`.run/` 只是 artifact output 位置，不是合同源。

- `handoff_contract.py` 拥有 zip layout、sidecar schema、task metadata schema、artifact manifest schema、validation-summary schema、`ProofClass`、validation row source contract、replay command contract 和 `EvidenceGraph`。
- `receipt_contract.py` 拥有 receipt exact schema、receipt `FieldSpec`、field proof class、receipt invariant、normalize 和 compare rules。
- `replay_evidence.py` 只从 repo root replay 并生成 canonical receipts；它不定义合同、不定义 allowed keys、不定义 validation rows、不定义 compare rules。
- package-smoke 是 `replay_evidence.py` 内部 executable replay 的一部分，不提供独立 receipt writer 入口。
- `verify_handoff_package.py` 只做 package-level orchestration 和 trust boundary。
- `governance_contract.py` 拥有 governance scan、source-tree scan、share-safety path rules 和 status tokens。

## 2. 正式 build entry

正式 handoff 只通过唯一 builder 命令生成：

```text
python .agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py --repo-root . --out-dir .run/handoffs --task-name <task-name> --base-ref <base-ref> --mode evidence --verify
```

builder 从当前 Git state 读取 branch 和 HEAD，要求 worktree clean，计算 `base = git merge-base HEAD <base-ref>`，自动生成 task metadata、machine receipts、validation summary、artifact manifest、`final-branch.bundle`、zip 和 external sidecar。

builder 只读取当前 Git state 和上述 CLI options；所有 machine evidence 都由 replay 自动生成。

## 3. Zip layout

正式 handoff zip 只允许 7 个 entries：

```text
artifact_manifest.json
final-branch.bundle
task_metadata.json
handoff_evidence/validation-summary.json
handoff_evidence/package-smoke-receipt.json
handoff_evidence/hard-scan-receipt.json
handoff_evidence/check-receipts.json
```

不允许 status-only 摘要、receipt index、generation/reconstruction 文件、human review outputs、raw logs、wheel、sdist、CSV、parquet、pickle、feather、arrow 或 private/generated data。

## 4. Sidecar

external sidecar 只是 zip envelope，只允许：

```text
schema_id
zip_filename
zip_sha256
```

sidecar 不重复 task metadata，不保存 branch/base/head，不保存 status summary，不自证 verifier 或 share-safety status。

## 5. Task metadata

`task_metadata.json` 只允许：

```text
schema_id
task_name
mode
branch
base
final_head
```

verifier 必须证明：

```text
final_head == bundle clone HEAD
base is ancestor of final_head
branch is the bundle clone target branch
```

## 6. Validation summary

`handoff_evidence/validation-summary.json` 只允许：

```text
schema_id
rows
```

每个 row 只允许：

```text
name
status
```

固定 rows：

```text
scripted_checks
package_smoke
profile_smoke
private_validation
baseline_compare
governance
code_map
```

来源固定：

```text
scripted_checks -> check-receipts.status
package_smoke -> package-smoke-receipt.status
profile_smoke -> check-receipts.profile-smoke.status
governance -> hard-scan-receipt.status
code_map -> check-receipts.code-map.status
private_validation -> NOT_RUN
baseline_compare -> NOT_RUN
```

`private_validation` 和 `baseline_compare` 在当前 machine handoff 中是 absence declaration，不生成 check-receipt command row。

## 7. Receipt invariants

`check-receipts.json` 只包含 `profile-smoke` 和 `code-map` 两个 command rows。每个 row 只允许 `schema_id`、`command_id`、`scope`、`command_argv`、`command`、`status`、`exit_code`、`counts`。check rows 不表达 `NOT_RUN` 或 `BLOCKED`；replay 失败就是 `FAIL`。

`package-smoke-receipt.json` 只保存 build/install/import/content_scan 的状态和固定 metadata；不保存 wheel/sdist 文件名、hash 或 content examples。

`hard-scan-receipt.json` 只保存 `schema_id`、`status`、`scanned_head`、`scanner_source_sha256` 和 `scan_runs`。每个 scan run 只保存 scan id、rule set、counts、result hash 和 replayable marker；不保存人工分类或 false-positive/deferred 计数。

human/advisory review 是独立人类审查输出，不进入 machine handoff，也不作为 machine validation `PASS` 的来源。

## 8. Verifier replay

verifier 从 zip 中取出 `final-branch.bundle`，clone task metadata 指定的 branch，检查 final head 和 base ancestor，然后在 clone 中调用 `replay_evidence.build_machine_receipts()`。packaged receipts 必须与 replayed receipts 的 normalized form 一致；packaged validation summary 必须与 replayed receipts 派生结果一致。
