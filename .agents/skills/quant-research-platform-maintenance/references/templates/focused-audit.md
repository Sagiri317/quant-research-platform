# 聚焦审查（Focused audit）

用于 `independent-review`。默认只读，不改代码。

## 开工前

```text
Task mode: independent-review
Base/head:
Context budget: small/medium
Review scope:
Write permission: no
```

## 审查顺序

1. 先看 `git status --short --branch`、`git diff --stat`、`git diff --name-only`。
2. 只读 changed files；不要先读全 repo。
3. 按 issue family 扫描：public surface、private safety、artifact truth、docs/status drift、owner layer drift、naming、script portability。
4. 命中很多时先汇总，不打开全部文件。
5. 输出 findings；不要直接修。

## 输出格式

```text
Finding ID:
Severity: blocker / major / minor / note
File/line or evidence:
Issue family:
Why it matters:
Suggested fix:
Confidence:
```

如果没有发现 blocker，也要说明哪些检查 `NOT_RUN`。
