# Handoff and Evidence

本文定义当前 `REVIEW_SAFE` machine handoff 合同。顶层文档只保留 owner
原则；具体 layout、字段和验证语义放在本文和可执行合同中。

## 工具 Owner

tracked handoff tooling 位于：

```text
.agents/skills/quant-research-platform-maintenance/scripts/handoff/
```

`.run/` 只是输出层，不定义 handoff 合同。

- `handoff_contract.py` 拥有 zip layout、sidecar schema、task metadata
  schema、artifact manifest schema、validation summary schema、`ProofClass`、
  validation row source contract 和 replay command contract。
- `receipt_contract.py` 拥有 exact receipt schema、receipt `FieldSpec`、
  receipt invariant、normalization 和 comparison。
- `replay_evidence.py` 执行 public machine replay，并构造 canonical
  receipts；hard-scan receipt 由
  `governance_contract.source_tree_scan_groups()` 的扫描结果构造。
- `verify_handoff_package.py` 是 package trust boundary。
- `governance_contract.py` 拥有 governance scan、source-tree scan、
  share-safety path rules 和 status tokens。

## Build Entry

正式 handoff 只能通过唯一 builder 命令生成：

```text
python .agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py --repo-root . --out-dir .run/handoffs --task-name <task-name> --base-ref <base-ref>
```

builder 从当前 Git state 读取 branch 和 HEAD，要求 worktree clean，计算
`base = git merge-base HEAD <base-ref>`，通过 replay 自动生成 machine
evidence，写入 bundle、receipts、validation summary、manifest、zip 和 external
sidecar，然后永远运行 verifier。verifier 不通过时，本次 build 不视为成功。

## Zip Layout

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

zip 不允许包含 status-only summary、receipt index、manual review output、raw
logs、raw stdout/stderr、wheel、sdist、CSV、parquet、pickle、feather、arrow、
vendor raw data、private data 或 generated private output。

## Sidecar

external sidecar 只是 zip envelope，只允许：

```text
schema_id
zip_filename
zip_sha256
```

sidecar 不重复 task metadata，不保存 status summary、verifier status 或
share-safety status。

## Task Metadata

`task_metadata.json` 只允许：

```text
schema_id
task_name
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

## Validation Summary

`handoff_evidence/validation-summary.json` 只允许 `schema_id` 和 `rows`。每个
row 只允许 `name` 和 `status`。

当前 rows 只保留 replay-backed 项：

```text
scripted_checks -> check-receipts.status
package_smoke -> package-smoke-receipt.status
profile_smoke -> check-receipts.profile-smoke.status
governance -> hard-scan-receipt.status
code_map -> check-receipts.code-map.status
```

## Receipt Invariants

`check-receipts.json` 只包含 `profile-smoke` 和 `code-map` 两个 command rows。
check rows 不表达 `NOT_RUN` 或 `BLOCKED`；replay 失败就是 `FAIL`。

`package-smoke-receipt.json` 只保存 build/install/import/content_scan 的状态和
固定 metadata，不保存 wheel 或 sdist 文件名、hash 或 content examples。

`hard-scan-receipt.json` 只保存 `schema_id`、`status`、`scanned_head`、
`scanner_source_sha256` 和 `scan_runs`。每个 scan run 只保存 `scan_id`、
`rule_set_id`、`match_count`、`actual_violation_count`、`result_sha256` 和
`replayable_from_bundle`。

## Verifier Replay

verifier 从 zip 中取出 `final-branch.bundle`，clone task metadata 指定的
branch，检查 final head 和 base ancestor，然后在 clone 中调用
`replay_evidence.build_machine_receipts()`。packaged receipts 必须与 replayed
receipts 的 normalized form 一致；packaged validation summary 必须与 replayed
receipts 派生结果一致。

human/advisory review 是独立人类审查输出，不进入 machine handoff，也不支撑
machine validation `PASS`。
