# Prompt Patterns

本文件提供 QRP 任务提示词模式。它不保存当前状态，不写具体 branch 或 commit。

## 1. 通用任务提示词骨架

```text
你是 QRP 的 Codex 执行者。
先运行 live git baseline，不继承旧状态。
任务目标：<目标>
允许范围：<paths>
禁止范围：<paths/actions>
必须验证：<commands>
最终报告：branch、HEAD、changed files、commands、status、remaining issues。
```

## 2. Docs cutover

```text
执行 QRP governance docs cutover。
目标是删除过期状态文档、合并重复事实源、保留 current contract。
不得恢复单独的状态文档或状态简报。
应用 patch 后必须修仍引用旧文档的治理脚本和测试。
```

## 3. Governance script fix

```text
修 verify_governance.py，使其支持 state-based active-doc scan。
clean tree 下也必须扫描 active docs。
checked_files 为空不能 PASS。
补 tests/agents/test_verify_governance.py。
```

## 4. Handoff package

```text
使用 .agents/skills/quant-research-platform-maintenance/scripts/handoff/ 下 tracked scripts 生成 handoff。
.run 仅作为 output/evidence layer。
报告 builder、verifier、source-tree scan、single verifier package-safety scan 和 handoff gate 的独立结果。
```

## 5. Audit-only

```text
只读审查，不改文件、不 commit、不 push。
必须给出 command transcript、scan coverage、file:line evidence、severity、impact path、fix plan。
```

## 6. Stop rule prompt

```text
如果任务需要访问 private data、修改 forbidden scope、恢复过期状态文档、或把 NOT_RUN/BLOCKED 写成 PASS，立即停止并报告。
```
