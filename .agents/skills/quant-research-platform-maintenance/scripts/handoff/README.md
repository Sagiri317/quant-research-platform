# Handoff Scripts

本目录承载 tracked QRP handoff tooling。handoff package 必须由这些脚本构建和
验证，不能从 `.run/` 输出反推合同。

## 文件

```text
handoff_contract.py
receipt_contract.py
replay_evidence.py
build_handoff_package.py
verify_handoff_package.py
self_test.py
README.md
```

## Contract Owners

- `handoff_contract.py` 拥有 zip layout、sidecar schema、task metadata
  schema、artifact manifest schema、validation summary schema、`ProofClass`、
  validation row source contract 和 replay command contract。
- `receipt_contract.py` 拥有 exact receipt schemas、receipt field specs、
  receipt invariants、normalization 和 replay comparison。
- `replay_evidence.py` 运行 public machine replay，并返回 canonical
  receipts；hard-scan receipt payload 从
  `governance_contract.source_tree_scan_groups()` 构造。它不定义 schema、
  allowed keys、proof classes、validation rows、replay commands 或 comparison
  rules。
- `verify_handoff_package.py` 是 package-level trust boundary。它读取
  sidecar，验证 zip envelope 和 exact member payloads，clone
  `final-branch.bundle`，检查 task metadata Git identity，在 clone 中 replay
  public receipts，比较 packaged receipts 和 replayed receipts，并比较 packaged
  validation summary 与 replay-derived summary。

## Build Entry

正式 build entry 是：

```text
python .agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py --repo-root . --out-dir .run/handoffs --task-name <task-name> --base-ref <base-ref>
```

builder 读取当前 Git state，要求 clean worktree，计算
`base = git merge-base HEAD <base-ref>`，写入 `final-branch.bundle`，运行
machine replay，写入 canonical receipts，派生 validation summary，写入 manifest、
zip 和 external sidecar，然后永远运行 verifier。verifier 不通过时，本次 build
不视为成功。

builder 只读取当前 Git state 和上述 CLI options；所有 machine evidence 都由
replay 自动生成。

## Package Layout

zip 必须只包含：

```text
artifact_manifest.json
final-branch.bundle
task_metadata.json
handoff_evidence/validation-summary.json
handoff_evidence/package-smoke-receipt.json
handoff_evidence/hard-scan-receipt.json
handoff_evidence/check-receipts.json
```

external sidecar 只包含：

```text
schema_id
zip_filename
zip_sha256
```

task metadata 只包含：

```text
schema_id
task_name
branch
base
final_head
```

verifier 从 `task_metadata.json` 读取 branch、base 和 final head，不从 sidecar
读取这些身份字段。

## Receipt Rules

Receipt JSON payloads 是 exact-schema compact machine receipts。它们不得包含 raw
stdout/stderr、raw grep output、完整 pytest logs、本地 install paths、wheel/sdist
文件或 private/generated data。

`hard-scan-receipt.json` 只包含 `schema_id`、`status`、`scanned_head`、
`scanner_source_sha256` 和 `scan_runs`。每个 scan run 只包含 `scan_id`、
`rule_set_id`、`match_count`、`actual_violation_count`、`result_sha256` 和
`replayable_from_bundle`。

validation summary 从 replayed receipts 派生，只包含：

```text
schema_id
rows
```

每个 row 只包含 `name` 和 `status`。当前只允许这些 replay-backed rows：

```text
scripted_checks
package_smoke
profile_smoke
governance
code_map
```

human/advisory review 是独立人类审查输出，不进入 machine handoff，也不支撑
machine validation `PASS`。
