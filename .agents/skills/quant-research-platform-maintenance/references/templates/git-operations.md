# Git 操作门（Git operations）

用于 commit、push、merge、branch cleanup。Git 操作不能顺手改业务代码。

## 基础检查

```text
git status --short --branch
git branch --show-current
git log --oneline -5
git diff --stat
git diff --name-only
```

## Commit / push

- 只 add 明确允许路径；不要 `git add .`。
- commit 前运行相关验证和 `git diff --check`。
- push 前确认目标 remote branch。
- 不 force push。

## Merge gate

- 先确认 base/head 和 handoff evidence；PR review 语境才使用 review evidence。
- 优先 fast-forward merge。
- 不在 merge gate 中修业务 bug。
- merge 到 `main` 必须单独获得用户明确批准。

## Final report

```text
branch
base/head
changed files
checks run
commit hash
push result
remaining risks
```
