# QRP 任务模板索引

先用 `task-modes.md` 选 mode，再只加载一个最匹配的 primary template。不要默认读取全部模板。

| Template | Mode | 用途 |
|---|---|---|
| `auto-task-router.md` | short / ambiguous prompts | 用户只写一两句话时先路由 |
| `focused-implementation.md` | `standard-implementation` | 有明确 owner layer 的实现/bugfix |
| `focused-audit.md` | `independent-review` | 只读审查，不改代码 |
| `goal-closure.md` | `goal-closure` | 长任务和 review-fix loop |
| `handoff-packaging.md` | `handoff-packaging` | review/handoff 包装 |
| `git-operations.md` | `merge-gate` / git task | commit、push、merge、branch cleanup |
| `docs-status-sync.md` | `docs-status-sync` | active docs/status/brief 同步 |
| `apply-only-validation.md` | `apply-only-validation` | 精确应用外部 patch/zip 并验证 |

短提示词例子：

```text
检查当前分支有没有 public API 扩张，只读 review。
同步当前 status 和 brief，只改治理文档，验证后提交。
用 .run 官方脚本生成 handoff 包，保持 public-safe。
应用这个 patch，验证没问题再 commit/push，不要重新设计。
```
