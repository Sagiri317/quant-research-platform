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
  `FieldSpec`、`allowed_keys()`、validation row source contract 和 replay
  command contract。
- `receipt_contract.py` 拥有 exact receipt schema specs、receipt invariants、
  normalization 和 replay comparison；它使用 `handoff_contract.py` 的共享
  `FieldSpec`，不定义第二套 field spec 类型或 allowed-key helper。
- `replay_evidence.py` 运行 public machine replay，并返回 canonical
  receipts；hard-scan receipt payload 从
  `governance_contract.source_tree_scan_groups()` 构造。它不定义 schema、
  allowed keys、proof classes、validation rows、replay commands 或 comparison
  rules。
- `verify_handoff_package.py` 是 package-level trust boundary。
  `verify_handoff_package_result()` 是唯一 package-level verifier function。
  它只做一次 zip inspection，并在需要 replay 时只 clone `final-branch.bundle`
  一次。standalone verifier CLI 保留，但不是额外 Python wrapper。未经用户明确
  批准，不保留 wrapper、alias、adapter 或 compatibility layer。verifier 读取
  sidecar，验证 zip envelope 和 exact member payloads，检查 task metadata Git
  identity，在同一个 clone 中执行 share-safety scan 和 public receipt replay，
  比较 packaged receipts 与 replayed receipts，并比较 packaged validation summary
  与 replay-derived summary。

## Build Entry

正式 build entry 是：

```text
python .agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py --repo-root . --out-dir .run/handoffs --task-name <task-name> --base-ref <base-ref>
```

builder 读取当前 Git state，要求 clean worktree，计算
`base = git merge-base HEAD <base-ref>`，写入 `final-branch.bundle`，运行
machine replay，写入 canonical receipts，派生 validation summary，写入 manifest、
zip 和 external sidecar，然后永远通过 `verify_handoff_package_result()` 运行
verifier。builder 不再运行独立 safety scan；share-safety status 来自同一个
verification result。verifier 不通过时，本次 build 不视为成功。

builder 只读取当前 Git state 和上述 CLI options；所有 machine evidence 都由
replay 自动生成。

## Package Contract

Package layout、sidecar schema、task metadata schema、artifact manifest schema、
validation summary schema、`ProofClass`、`FieldSpec`、`allowed_keys()`、
validation row source contract 和 replay command contract 由
`handoff_contract.py` 定义。Receipt schemas、invariants、normalization 和
comparison 由 `receipt_contract.py` 定义。

本目录文档不得重复完整 zip entries、sidecar keys、task metadata keys、
validation rows 或 receipt field lists。Verifier 从 task metadata 读取 branch、
base 和 final head，不从 sidecar 读取这些身份字段。

## Receipt Rules

Receipt JSON payloads 是 exact-schema compact machine receipts。它们不得包含 raw
stdout/stderr、raw grep output、完整 pytest logs、本地 install paths、wheel/sdist
文件或 private/generated data。

Validation summary 从 replayed receipts 派生。Exact rows 和 fields 由
`handoff_contract.py` 与 `receipt_contract.py` 拥有，不由本 README 定义。

human/advisory review 是独立人类审查输出，不进入 machine handoff，也不支撑
machine validation `PASS`。

## Test Performance Path

mutation tests 使用 replay-compare fast path：先做 sidecar hash、zip layout、
artifact manifest、payload schema 和 validation-summary-against-receipts 校验，
再用 session-scoped cached replay receipts 调用
`compare_replayed_handoff_payloads()`。只有 valid package full verifier smoke、
Git identity mutation 和 sidecar hash mutation 保留完整 bundle verifier 路径。
