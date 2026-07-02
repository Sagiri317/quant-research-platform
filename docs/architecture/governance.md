# Governance

QRP 的治理目标是让 agent 能在最少上下文下正确执行，并通过可验证 gate 保护当前合同和 REVIEW_SAFE 边界。

## 核心规则

1. live git baseline 是现场事实入口。
2. 文档不保存当前状态。
3. `.agents` 是 tracked tooling source。
4. `.run` 是 artifact/evidence layer。
5. `.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 是内部治理规则事实源。
6. `PASS` 必须绑定实际命令或 artifact。
7. current-contract-only 是默认策略。
8. 边界回归由 owner-layer negative tests 和精确 allowlist 控制。
9. QRP-owned JSON 必须使用不带版本后缀的 `schema_id`。
10. 状态聚合唯一 owner 是 `policy/check_status.py`。

## Agent 文件分层

```text
AGENTS.md      根规则，短而硬
SKILL.md       skill 入口和触发规则
references/    按需读取的长规则
scripts/       可执行治理工具
schemas/       机器合同
templates/     机器输入模板
```

## 禁止状态文档

仓库不保存单独的状态文档或状态简报。任何需要“当前状态”的任务都必须先运行 git baseline。

## Open issues

`open-issues.md` 只保存 current open 和 deferred 问题。

## Gate 分层

- diff gate：检查 patch 格式。
- governance gate：检查 active docs、privacy、current-truth facts、status token。
- JSON gate：检查 QRP-owned JSON 的 object 顶层与 `schema_id`。
- naming gate：检查 current-contract path、文件/路径名、Python symbol 和静态字符串拼接。
- code-map gate：检查代码地图与当前合同一致。
- handoff gate：检查 builder、verifier、source-tree scan 和 share-safety。
- handoff evidence gate：检查 required evidence 存在、结构完整且不是 status-only。
- internal API gate：检查 owner-local internal API 和 private module 边界。
- test placement gate：检查 tests 位于对应 owner 目录。
- test gate：检查当前合同测试。

## 返工防护

如果 Codex 或其他 agent 需要越过 current-contract-only 边界，必须停止并要求用户确认。

active-doc scan 必须覆盖运行结果命名、运行级路径、JSON identity、compare 硬编码和状态 owner；allowlist 必须精确到规则和负向测试文件。scan classification 的细则由 maintenance references、governance contract 和 verifier tests 拥有。
