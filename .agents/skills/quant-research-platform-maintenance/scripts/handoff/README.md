# Handoff Scripts

本目录保存 tracked handoff tooling。所有 handoff 构建和验证必须使用这里的脚本，不得从 `.run/` 中复制或推断工具逻辑。治理规则从 `../governance_contract.py` 读取；该文件是共享规则事实源，不是 `.run/` 产物。

## 文件

```text
build_handoff_package.py
verify_handoff_package.py
self_test.py
```

## 分层

- builder 只负责生成 bundle、manifest、sidecar、task metadata 和 evidence，并调用 verifier。
- verifier 是唯一 handoff/package verifier，负责 bundle 合同、manifest/hash、zip member、sidecar、source-tree scan 和 share-safety 检查。
- source-tree scan 规则来自 `governance_contract.py`。
- share safety 规则来自 `src/quant_research_platform/policy/share_safety.py`。

这些 gate 互不替代。任何报告必须分别写出状态。

## 输出

handoff 输出应写到 `.run/` 或用户指定目录。`.run/` 是 output layer，不是工具事实源。

`artifact_manifest.json` 是 package manifest anchor。它必须存在于 zip 根部，
由 sidecar 记录 hash，但不列入自己的 `entries`；`entries` 只覆盖 manifest
锚点之外的 package 成员。

handoff package 内的 `final-branch.bundle` 应可用以下形式重建分支：

```text
git clone -b <branch-from-sidecar> final-branch.bundle review-repo
```

When `final-branch.bundle` is present, the builder records bundle verify,
list-heads, and the exact branch clone evidence in
`handoff_evidence/bundle-reconstruction.json`; the verifier checks that
evidence and still runs its own bundle reconstruction checks.

handoff 合同只由本目录 tracked scripts、schemas 和 templates 定义。

## 运行纪律

运行前先确认 live git baseline。生成 bundle 后记录 hash、manifest、sidecar、source-tree scan、single verifier result、git bundle verify 和 list-heads。未运行项写 `NOT_RUN`，环境或超时阻塞写 `BLOCKED`。
