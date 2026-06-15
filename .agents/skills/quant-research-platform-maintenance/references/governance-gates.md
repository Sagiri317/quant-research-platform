# Governance Gates / 治理关卡

| Gate | 目的 | 触发任务类型 | 层级 | 防止的返工 | 可机器检查 |
|---|---|---|---|---|---|
| Task Router Gate | 先判定任务模式，避免把 review/package/merge/governance 误当实现 | all | `AGENTS.md`, `task-modes.md`, router template | 目标混淆、scope drift | partial |
| Scope Gate | 明确 allowed/forbidden scope、branch、文件范围 | medium+ | templates/manual checklist | 越界改业务代码、误动 main | partial |
| Context Gate | 最小上下文加载，避免读错/读太多 | all | `context-routing.md` | token 浪费、漏读关键文件 | partial |
| Owner Layer Gate | 先找 owner layer，再改 owner layer | implementation/P2/P3 | `owner-layer-routing.md` | 跨层大改、顺手修错层 | partial |
| Blast Radius Gate | 评估 public/private/cross-layer 影响半径 | implementation/governance/release | template/manual | 小任务变架构重写 | partial |
| Red Test Gate | 行为变更先有失败用例或缺口说明 | implementation | template/tests | 修了表面没封合同 | yes/partial |
| Same Issue Family Scan Gate | 按问题家族搜索相似问题 | implementation/review/goal | template + `rg` | 同类 bug 漏修 | partial |
| Artifact Truth Gate | 保留 PASS/FAIL/NOT_RUN 真实性 | handoff/evidence/release | contracts + script | 未跑写 PASS、FAIL 被掩盖 | partial |
| Public Surface Gate | 阻止 public API/CLI/registry/store/facade 扩张 | implementation/release | AGENTS + script + tests | 无意新增公共面 | partial |
| Private Safety Gate | 阻止 private data 泄露 | artifact/report/handoff/API/CLI | AGENTS + script + tests | row-level data 泄露 | yes/partial |
| Memory/Performance Awareness Gate | 防止热路径回归、重复 IO、内存爆炸 | runtime/storage/evaluation/model | template/manual/tests | 性能返工 | partial |
| Docs/Status Sync Gate | 只同步 active docs，避免 stale current truth | medium+/docs/release | template/script | docs 与代码/status 冲突 | partial |
| `.run` Tooling Gate | 正确使用 `.run` 官方工具，不让 handoff 驱动架构 | handoff/goal/review | template + package script | 包格式、安全、证据缺失 | partial |
| Script Portability Gate | 脚本跨 Win11/macOS/Linux，避免 ps1-only | governance/scripts/handoff | script + review | 本地不可运行 | yes/partial |
| Handoff Gate | 打包 public-safe review bundle/git bundle/manifest | handoff/merge/release | `.run` + safety script | reviewer 无法复现或包泄露 | partial |
| Final Truthfulness Gate | final response 列出 changed/not changed/tests/not run/risks | all | template/manual | 虚假完成、遗漏风险 | no/partial |

## Gate placement rule

- Hard prohibitions live in `AGENTS.md`.
- Mode selection and context budgets live in `.agents` references/templates.
- Repetitive mechanical checks live in `.agents/skills/**/scripts` or tests.
- Long-term rationale lives in `docs/architecture/`.
- `.run` remains local evidence/tooling layer, not source architecture.
- hooks/subagents/MCP/plugins remain draft-only until a clear repeated workflow justifies activation.
