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

1. builder：构建 bundle，写入必要 manifest、sidecar、metadata 和 evidence，并调用 verifier。
2. verifier：唯一 handoff/package verifier，验证 bundle、manifest/hash、zip member、sidecar、source-tree scan 和 share-safety。
3. source-tree scan：扫描当前源码树是否符合 current-contract-only 规则；规则来自 `governance_contract.py`。
4. policy share safety：文本、JSON、路径和 zip member 的共享安全规则来自 `policy/share_safety.py`。

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

正式 handoff 的 REVIEW_SAFE evidence 必须可由 verifier 审查：

status-only 摘要不能作为完整 evidence；具体判定由 tracked verifier 和 tests 拥有。

## 4. Sidecar 规则

sidecar 是机器可读 evidence summary，不是业务结论本身。sidecar PASS 只表示 sidecar 对应检查通过，不代表 full pytest、local private validation 或 business correctness。

## 5. Git Bundle

handoff package 和 git bundle 是不同概念。handoff package 可以包含 REVIEW_SAFE reports、logs、manifest 和 evidence；git bundle 应能重建当前分支仓库状态。生成和验证时应记录 bundle hash、list-heads 和推荐 clone command。
