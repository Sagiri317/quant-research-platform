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
README.md
```

## Contract Owners

- `handoff_contract.py` 拥有 zip layout、sidecar schema、task metadata
  schema、artifact manifest schema、validation summary schema、`ProofClass`、
  `FieldSpec`、exact schema validation、validation row source contract 和 replay
  command contract。
- `receipt_contract.py` 拥有 exact receipt schema specs、receipt invariants、
  normalization 和 replay comparison；它使用 `handoff_contract.py` 的共享
  `FieldSpec`，不定义第二套 field spec 类型或 allowed-key helper。
- `replay_evidence.py` 运行 public machine replay，并返回 canonical
  receipts；hard-scan receipt payload 从
  `governance_contract.source_tree_scan_groups()` 构造。它不定义 schema、
  allowed keys、proof classes、validation rows、replay commands 或 comparison
  rules。package smoke 只通过 wheel 解压和 `PYTHONPATH` import smoke 构造，
  不执行安装语义。
- `verify_handoff_package.py` 是 package-level trust boundary。
  `verify_handoff_package_result()` 是唯一 package-level verifier function。
  它使用 single `HandoffPackagePayload` load 和 package payload validation，并在
  需要 replay 时只 clone `final-branch.bundle` 一次。standalone verifier CLI
  保留为 verifier 的直接入口。未经用户明确批准，不新增或保留 wrapper、alias、
  adapter 或 compatibility layer。verifier 读取 sidecar，验证 exact member
  payloads，检查 task metadata Git identity，在同一个 clone 中执行 share-safety
  scan 和 public receipt replay，比较 packaged receipts 与 replayed receipts，并
  比较 packaged validation summary 与 replay-derived summary。

## Build Entry

正式 build entry 是：

```text
python .agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py --repo-root . --out-dir .run/handoffs --task-name <task-name> --base-ref <base-ref>
```

builder 读取当前 Git state，要求 clean worktree，计算
`base = git merge-base HEAD <base-ref>`，写入 `final-branch.bundle`，从该 bundle
clone 运行 machine replay，写入 canonical receipts，派生 validation summary，写入
manifest、zip 和 external sidecar，然后通过 package payload validation 检查
schema、hash 和 share-safety。builder 不运行独立于 package flow 的 safety scan；share-safety
status 来自同一个 package payload validation result。检查不通过时，本次 build
不视为成功。

standalone verifier 仍是完整 bundle replay 和 receipt comparison 的正式入口；它从
`final-branch.bundle` clone，一次性复用 `SourceTreeScanInput` 驱动的 source-tree
scan 结果执行 share-safety 和 hard-scan receipt replay，并比较 packaged receipts
与 replayed receipts。

builder 只读取当前 Git state 和上述 CLI options；所有 machine evidence 都由
replay 自动生成。

## Package Contract

Package layout、sidecar schema、task metadata schema、artifact manifest schema、
validation summary schema、`ProofClass`、`FieldSpec`、exact schema validation、
validation row source contract 和 replay command contract 由
`handoff_contract.py` 定义。Receipt schemas、invariants、normalization 和
comparison 由 `receipt_contract.py` 定义。

本目录文档只描述 owner layer、scope boundary 和 acceptance gate。Verifier 从
task metadata 读取 branch、base 和 final head。

## Receipt Rules

Receipt JSON payloads 是 exact-schema compact machine receipts。Review-safe
payload 只保存 replayable status、schema、count、hash 和 command metadata；
不得包含 raw stdout/stderr、raw grep output、完整 pytest logs、本地临时路径、
wheel/sdist 文件或 private/generated data。

Validation summary 从 replayed receipts 派生。Exact rows 和 fields 由
`handoff_contract.py` 与 `receipt_contract.py` 拥有，不由本 README 定义。
`profile_smoke` row 只代表正式 profile runner 的 `profile-smoke` profile。
command rows 必须记录可校验的 `duration_seconds`，但 replay comparison 不依赖
动态耗时相等。

human/advisory review 是独立人类审查输出，不进入 machine handoff，也不支撑
machine validation `PASS`。

## Test Performance Path

mutation tests 使用 replay-compare fast path：先做 sidecar hash、zip layout、
artifact manifest、payload schema 和 validation-summary-against-receipts 校验，
再用 session-scoped cached replay receipts 调用
`validate_handoff_package_payload()` 与 `compare_replayed_handoff_payloads()`。
只有 valid package full verifier smoke、Git identity mutation 和 sidecar hash
mutation 保留完整 bundle verifier 路径。
