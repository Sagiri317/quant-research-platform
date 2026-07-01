# Handoff and Evidence

本文定义 handoff 和 evidence 的当前合同。

## 1. 工具位置

tracked handoff tooling 位于：

```text
.agents/skills/quant-research-platform-maintenance/scripts/handoff/
```

`.run/` 只是 artifact output 位置。不要从 `.run/` 反推工具合同。

## 2. Handoff 分层

handoff 至少包含：

1. check runner：实际执行检查并生成 compact receipt。
2. builder：只读取 task metadata、receipts-dir，构建 `final-branch.bundle`、manifest、sidecar 和 receipt evidence。
3. verifier：唯一 handoff/package verifier，校验 bundle、manifest/hash、zip member、sidecar、receipt schema、source-tree scan 和 share-safety。
4. hard-scan replay：可从 `final-branch.bundle` 重放的检查只保存 receipt，verifier 在 bundle clone 中重放并比对 count/hash。
5. policy share safety：文本、JSON、路径和 zip member 的共享安全规则来自 `policy/share_safety.py`。

这些结果不能互相替代。builder 应 fail fast，不能先写出非法 artifact 再依赖 verifier 补救。handoff gate 只依据本轮实际运行的独立检查。

## 3. Evidence 规则

REVIEW_SAFE evidence 可以包含：

- command
- status token
- exit result
- counts
- schema hash
- artifact hash
- REVIEW_SAFE path structure

REVIEW_SAFE evidence 不得包含：

- private row-level data
- vendor marker
- local absolute path
- stock-code sample
- raw traceback
- private-derived sample

LOCAL_ONLY outputs 默认不进入 handoff。原因是共享边界、授权边界、体积边界和本地 path/provider/source 风险，而不是因为所有结果本身都必然是隐私。

handoff 可以纳入完整 `run.json`，但必须校验 `schema_id` 并通过统一 safety verifier。handoff zip 内机器 evidence 统一放在 `handoff_evidence/`。

正式 handoff 的 REVIEW_SAFE evidence 必须可由 verifier 审查。Codex 不应手写或搬运大段 handoff evidence JSON；检查由脚本运行，receipt 由脚本生成，builder 只打包 compact receipts。

status-only 摘要不能作为完整 evidence；具体判定由 tracked verifier 和 tests 拥有。

正式 handoff 必须包含以下机器 evidence：

```text
handoff_evidence/package-smoke-receipt.json
handoff_evidence/hard-scan-receipt.json
handoff_evidence/check-receipts.json
handoff_evidence/private-smoke-status.json
handoff_evidence/generation.json
handoff_evidence/source-tree-scan.json
handoff_evidence/bundle-reconstruction.json
```

`advisory-review-receipt.json` 是 optional。只有本轮声明 advisory/subagent review `PASS` 时才必须存在；未运行时 sidecar 写 `NOT_RUN` 或不声明 PASS。

`package-smoke-receipt.json` 只记录 wheel/sdist 文件名与 hash、临时 no-deps install/import 状态、content scan 状态和 forbidden count；wheel、sdist、临时 install target 不进入 handoff。`hard-scan-receipt.json` 不保存 raw grep output 或完整命中，只保存 scan id、rule set、count、result hash 和必要 classified exceptions。不可重放的 private/local 检查只保存 `private-smoke-status.json` 这种 REVIEW_SAFE 状态摘要，不保存 row-level data 或本地 raw artifact。

handoff 不打包 raw grep output、full pytest log、完整 stdout/stderr、完整 subagent transcript、完整文件清单、wheel/sdist、本地临时安装目录、`.run/local/**` 或 private/generated data。

builder 必须拒绝 task JSON 中的 `handoff_evidence`、`handoff_markdown_evidence` 和 `validation_evidence`。`validation_evidence.json` 若存在于包内，只能由 builder 从 receipts 派生，不由 task 手写。

## 4. Sidecar 规则

sidecar 是机器可读 evidence summary，不是业务结论本身。sidecar PASS 只表示 sidecar 对应检查通过，不代表 full pytest、local private validation 或 business correctness。

sidecar status 必须拆分 `package_smoke`、`profile_smoke`、`private_smoke_run_level`、`private_smoke_child_checks`、`baseline_compare`、`handoff_verify`、`handoff_safety_scan` 等状态，不得使用 generic `smoke=PASS` 混淆 package/profile/private/compare。

## 5. Git Bundle

handoff package 和 git bundle 是不同概念。handoff package 可以包含 REVIEW_SAFE reports、logs、manifest 和 evidence；git bundle 应能重建当前分支仓库状态。生成和验证时应记录 bundle hash、list-heads 和推荐 clone command。

verifier 必须从 `final-branch.bundle` clone 一次并重放可重放检查，例如 hard scan；不可重放检查只校验 review-safe receipt schema 和状态真实性。已退役的状态文件不应恢复为 handoff 状态源。
