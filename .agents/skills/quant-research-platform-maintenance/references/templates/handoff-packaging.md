# 交接打包（Handoff packaging）

用于 handoff package。目标是 public-safe、可复现、证据真实；review bundle 只作为 historical / migrated 旧称出现。

## 执行规则

1. 使用 repo/local `.run` 官方 handoff 工具；不要临时拼包。
2. 先读 `.run` summary/package rules/manifest，不读 raw logs。
3. Git bundle 应放入 handoff package（如果项目规则要求），但 bundle 不得包含 private data。
4. Package manifest 必须真实记录 `PASS/FAIL/NOT_RUN/BLOCKED`。
5. 不把 handoff 结构反向写成架构规则。

## 包安全

运行：

```text
python .agents/skills/quant-research-platform-maintenance/scripts/package_safety_check.py <bundle-or-dir> --json
```

禁止包含 parquet/csv/pkl/pickle/feather/arrow/orc/db/sqlite、raw vendor data、row-level private data、credentials、tokens、本地绝对路径、旧 handoff zip/json。

## Final report

```text
package path
manifest path
package safety: PASS/FAIL
checks run / not run
private-safe evidence summary
open issues
```
