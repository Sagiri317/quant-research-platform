# Handoff and Evidence

本文定义当前 `REVIEW_SAFE` machine handoff 的 owner map 和执行边界。
具体 layout、字段和验证语义只由可执行合同与 tests 定义。

## 工具 Owner

tracked handoff tooling 位于：

```text
.agents/skills/quant-research-platform-maintenance/scripts/handoff/
```

`.run/` 只是输出层，不定义 handoff 合同。

- `handoff_contract.py` 拥有 zip layout、sidecar schema、task metadata
  schema、artifact manifest schema、validation summary schema、`ProofClass`、
  `FieldSpec`、`allowed_keys()`、validation row source contract 和 replay
  command contract。
- `receipt_contract.py` 拥有 exact receipt schema specs、receipt invariant、
  normalization 和 comparison。receipt specs 使用 `handoff_contract.py` 的共享
  `FieldSpec`；receipt 层不得定义第二套 `FieldSpec` 或 allowed-key helper。
- `replay_evidence.py` 执行 public machine replay，并构造 canonical
  receipts；hard-scan receipt 由
  `governance_contract.source_tree_scan_groups()` 的扫描结果构造。
  package smoke 使用 conda-managed non-isolated build：`python -m build --no-isolation --sdist --wheel --outdir <dist>`；构建后端依赖由 `environment.yml` 显式提供。package smoke 只通过 wheel 解压和 `PYTHONPATH` import smoke 构造，不执行安装。
- `verify_handoff_package.py` 是 package trust boundary。
  `verify_handoff_package_result()` 是唯一 package-level verifier function。
  verifier result path 只做一次 zip inspection，且需要 replay 时只 clone
  `final-branch.bundle` 一次。standalone verifier CLI 保留，但不是额外 Python
  wrapper。未经用户明确批准，不新增或保留 wrapper、alias、adapter 或
  compatibility layer。
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

`profile_smoke` validation row 只代表正式 profile runner 的 `profile-smoke`
profile；validation summary 必须从 packaged/replayed receipts 派生，不允许手写
`PASS`。

正式 handoff success 的 validation 语义是 mandatory validation rows 全部为
`PASS`。`FAIL`、`NOT_RUN`、`BLOCKED`、缺失、重复或未知状态都必须让 builder
和 verifier 失败；package layout 或 receipt 自洽不能替代 mandatory rows 全
PASS。validation summary 的顶层 status 只由 rows 聚合派生。

## 合同边界

`handoff_contract.py` 拥有 package layout、sidecar schema、task metadata
schema、artifact manifest schema、validation summary schema、`ProofClass`、
`FieldSpec`、`allowed_keys()`、validation row source contract 和 replay command
contract。

`receipt_contract.py` 拥有 receipt schemas、invariants、normalization 和
comparison。

`replay_evidence.py` 拥有 machine replay 和 canonical receipt construction。

`verify_handoff_package.py` 拥有 verifier result path、zip inspection primitive、
bundle replay primitive、replay comparison 和 clone tree share-safety scan。

文档只描述 owner layer、scope boundary 和 acceptance gate；字段事实源使用上述可执行 owner modules。

## Verifier Replay

verifier 从 zip 中取出 `final-branch.bundle`，clone task metadata 指定的
branch，检查 final head 和 base ancestor，然后在同一个 clone 中执行
share-safety scan 和 `replay_evidence.build_machine_receipts()`。packaged receipts
必须与 replayed receipts 的 normalized form 一致；packaged validation summary
必须与 replayed receipts 派生结果一致。payload/schema mutation tests 使用
`compare_replayed_handoff_payloads()` 的 fast path；完整 bundle replay 保留给
valid package smoke 和 Git identity 变异。

receipt command rows 必须记录可校验的 `duration_seconds`。动态耗时只用于性能和
运行事实，不作为 replay equality 的稳定语义。

human/advisory review 是独立人类审查输出，不进入 machine handoff，也不支撑
machine validation `PASS`。
