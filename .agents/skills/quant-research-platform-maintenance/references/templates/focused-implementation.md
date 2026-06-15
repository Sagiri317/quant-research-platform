# 聚焦实现（Focused implementation）

用于 `standard-implementation`。目标是局部、可验证、一遍过，不做跨层顺手重构。

## 开工前

```text
Task mode: standard-implementation
Owner layer:
Context budget: medium
Allowed files:
Forbidden files:
Public surface impact: none / requires approval
Private data impact: none / requires approval
```

## 执行规则

1. 先读 `AGENTS.md`、current status、repo-map、owner-layer routing。
2. 只读 owner layer 相关代码和 tests。
3. 如果需要 public API/CLI、registry、store、facade、artifact type，停止并请求批准。
4. 先写或定位 red test（红测），证明问题存在；无法做 red test 时说明原因。
5. 用 `rg` 做 same issue family scan，只打开相关文件。
6. 最小实现，删除被替代路径，不新增 legacy shim。
7. 同步必要 docs/status，但不要大改无关文档。

## 验证

```text
targeted pytest for touched owner layer
ruff / py_compile if Python files changed
git diff --check
public surface check if interfaces/contracts touched
private safety check if data/report/package touched
```

未运行检查写 `NOT_RUN` 或 `BLOCKED`。
