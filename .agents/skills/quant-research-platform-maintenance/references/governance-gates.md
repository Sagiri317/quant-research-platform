# 治理门（Governance gates）

中等及以上任务必须选择相关 gates。Gate 是执行前/执行中/执行后的硬检查，不是口号。

| Gate | 目的 | 触发任务 | 层级 | 是否机器可查 |
|---|---|---|---|---|
| Task Router | 先判断 mode，避免短提示词误路由 | 全部任务 | `AGENTS.md` + `task-modes.md` | 部分 |
| Scope | 明确允许/禁止改动 | 全部写任务 | template | 部分 |
| Context | 控制读取预算 | 全部任务 | `token-economy.md` | 部分 |
| Owner Layer | 防止跨层顺手修 | implementation/review | `owner-layer-routing.md` | 部分 |
| Blast Radius | 声明影响面 | medium+ | template | 人工 |
| Red Test | 先证明问题存在 | bugfix/contract | template/tests | 是 |
| Same Issue Family Scan | 防止同族漏修 | review/fix | `review-fix-loop.md` | 部分 |
| Artifact Truth | 防止 false PASS | reports/handoff/tests | verification docs/scripts | 部分 |
| Public Surface | 阻止 API/CLI 扩张 | interfaces/contracts | scripts/review | 部分 |
| Private Safety | 阻止私有数据泄露 | package/report/data | scripts/review | 部分 |
| Memory/Performance Awareness | 防止资源回退 | runtime/data/perf | code facts/tests | 部分 |
| Docs/Status Sync | 防 stale docs | medium+ | docs-status template | 部分 |
| `.run` Tooling | 正确用 handoff 工具 | handoff | `.run` + template | 部分 |
| Script Portability | 保证 Win11/macOS/Linux | scripts | script-portability | 部分 |
| Handoff | 保证 review 可复现 | handoff/release | template/scripts | 部分 |
| Final Truthfulness | 最终报告真实 | 全部任务 | final checklist | 人工+脚本 |

## Gate 输出格式

每个被触发的 gate 在 final report 中只需短写：

```text
Gate: PASS/FAIL/NOT_RUN/BLOCKED - evidence or reason
```

不要粘贴长日志。失败时贴最小定位证据。
