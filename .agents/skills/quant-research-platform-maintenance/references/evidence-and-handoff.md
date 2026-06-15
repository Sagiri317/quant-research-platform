# 证据与交接（Evidence and handoff）

本文件用于 review bundle、handoff package、public-safe evidence 和 `.run` 工具任务。

## `.run` 的定位

`.run/` 是 ignored local workflow/evidence/tooling layer（本地工作流/证据/工具层）。它可以证明任务做过什么，但不能反向定义架构。架构事实仍以代码、contracts、tests、active docs 为准。

## 读取顺序

1. 先读 `.run` 的 README、package rules、summary、manifest、index。
2. 再读 safety report、verification summary、comparison summary。
3. 只有 summary 指向失败或缺口时，才读相关 raw log 的最小片段。
4. 不读 row-level private data、raw vendor data、旧 handoff zip/json。

## package safety（包安全）

Review/handoff 包默认禁止：

```text
.parquet .csv .pkl .pickle .feather .arrow .orc .db .sqlite .sqlite3
raw vendor data
row-level private data
credentials / tokens / auth.json / .env
local absolute paths
old handoff zips/json unless sanitized summary is explicitly allowed
```

## Git bundle 与 review bundle

Git bundle 是可复现当前分支的 repo 状态；review bundle 是证据包。若任务要求 handoff，Git bundle 应放在 review bundle 内，但不得包含 private data。

## truthfulness（真实性）

Manifest 必须区分 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。不能为了显得完整把缺失验证写成通过。
