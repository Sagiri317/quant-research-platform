# 审查—修复闭环（Review-fix loop）

本文件用于减少 review 后返工。核心是把问题按 issue family（问题族）处理，不只修一个点。

## 流程

1. 从 review findings 提取 issue family，而不是逐条孤立修。
2. 每个 issue family 先定位 owner layer。
3. 用 `rg -n` 做同族扫描，先看命中摘要。
4. 只打开最相关文件，判断是否同类问题。
5. 修复后补 red test 或 regression check。
6. final report 写清：修了哪些 family，哪些 NOT_RUN/BLOCKED，哪些登记为 open issue。

## 常见 issue family

```text
public API/CLI expansion
private data leakage
artifact truthfulness
stale docs/status
owner layer drift
naming inconsistency
legacy alias / compatibility shim
script portability
run package safety
memory/performance regression
```

## Subagent 使用

复杂 review 可让 subagent read-only 检查单一 issue family。每个 subagent 必须有范围、输出上限和关闭要求；main agent 负责最终整合，不让 subagent 直接 commit/push。
