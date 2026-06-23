# Handoff and Evidence

本文件定义 handoff 和 evidence 的当前合同。

## 1. 工具位置

tracked handoff tooling 位于：

```text
.agents/skills/quant-research-platform-maintenance/scripts/handoff/
```

`.run/` 只是 artifact output 位置。不要从 `.run/` 反推工具合同。

## 2. Handoff 分层

handoff 至少分为四层：

1. builder：构建 bundle，写入必要 manifest、sidecar、metadata 和 evidence。
2. verifier：独立验证 bundle 结构和合同。
3. source-tree scan：扫描当前源码树是否含 forbidden surface。
4. package safety：检查 public-safe、路径安全、private marker 和 forbidden content。

四层结果不能互相替代。builder 应 fail fast，不能先写出非法 artifact 再依赖 verifier 补救。

## 3. Evidence 规则

public-safe evidence 可以包含：

- command
- status token
- exit result
- counts
- schema hash
- artifact hash
- non-private path structure

public-safe evidence 不得包含：

- private row-level data
- vendor marker
- local absolute path
- stock-code sample
- raw traceback
- private-derived sample

## 4. Sidecar 规则

sidecar 是机器可读 evidence summary，不是业务结论本身。sidecar PASS 只表示 sidecar 对应检查通过，不代表 full pytest、private validation 或 business correctness。

## 5. Git bundle

review bundle 和 git bundle 是不同概念。review bundle 可以包含 reports、logs、manifest 和 evidence；git bundle 应能重建当前分支仓库状态。生成和验证时应记录 bundle hash、list-heads 和推荐 clone command。

## 6. 历史 evidence

历史阶段 evidence 不保存在 active docs 中。需要追溯时使用 git history 或旧 handoff artifacts，不让历史 reports 进入当前 agent context。
