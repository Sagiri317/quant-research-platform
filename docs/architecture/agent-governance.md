# 智能体治理（Agent governance）

本文定义 QRP 的 agent governance layer（智能体治理层）。它不是日常 prompt；日常入口是 `AGENTS.md` 和 `.agents/skills/quant-research-platform-maintenance/SKILL.md`。

## 目标

治理层必须让用户用很少的提示词也能让 Codex / GPT / Claude 精准命中任务、自动遵守原则、少读上下文、少返工、一遍过。它服务以下长期原则：Semantic closure（语义收口）、Naming first-class（命名一等）、Replace over parallel（替代优先）、Minimal public API（最小公共接口）、Evidence over claims（证据优先）、Private-data safety（私有数据安全）、Layer-local change（局部改动）。

## 分层

| 层 | 职责 | 读取时机 |
|---|---|---|
| `AGENTS.md` | 硬边界、任务路由、停止条件、路径权威 | 每次任务 |
| `.agents/briefs/` | 当前事实、用户原则、短 repo map | 按需 |
| `SKILL.md` | QRP 操作路由和加载顺序 | QRP 任务触发时 |
| `references/` | task modes、token economy、gates、owner routing、外部工具共存规则 | 按任务类型 |
| `templates/` | mode-specific execution checklist（模式专用执行清单） | 一次只读一个 |
| `scripts/` | 确定性检查脚本 | 验证阶段 |
| `docs/architecture/agent-governance*.md` | 长期解释和设计取舍 | 治理/架构 review |
| `.run` | 本地 workflow/evidence/tooling | handoff/evidence 任务 |
| hooks/MCP/plugins/subagents config | draft-only optional adapters | 单独批准后 |

## 路径权威

- `.agents/` 是项目治理根。
- global `.codex/` 只作为用户显式提供的只读证据源。
- repo-local `.codex/` 默认不创建。
- 外部 MCP、plugins、global skills 可辅助，但不能覆盖 QRP 规则。

## Token economy

治理层采用 progressive disclosure（渐进式披露）：短入口、短 brief、按需 references、一个 mode 一个 template、证据先读 summary。`AGENTS.md` 不写百科，避免触发 Codex `project_doc_max_bytes` 限制和上下文污染。

## 一遍过机制

每个中等以上任务都必须选择相关 gates：Task Router、Scope、Context、Owner Layer、Blast Radius、Red Test、Same Issue Family Scan、Artifact Truth、Public Surface、Private Safety、Memory/Performance Awareness、Docs/Status Sync、`.run` Tooling、Script Portability、Handoff、Final Truthfulness。
