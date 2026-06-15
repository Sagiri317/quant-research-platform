---
name: quant-research-platform-maintenance
description: 用于 QRP / quant-research-platform 仓库内的治理、维护、任务路由、实现边界、审查、handoff 包装、merge gate、private-safety、公有 API/CLI 合同检查、owner-layer routing，以及 Codex/GPT/Claude 工作流纪律。不要用于 live trading、deployment、vendor login、广泛 private data access 或外部 API 集成。
---

# QRP 维护与治理 skill（Quant Research Platform Maintenance）

本 skill 是 QRP 的操作路由层。它的目标是让用户用很短的提示词也能让 Codex 精准命中任务、遵守长期原则、少读上下文、少返工、一遍过。不要把本文件扩写成百科；详细规则按需进入 `references/`。

## 加载顺序

1. 先读 repo 根目录 `AGENTS.md`。
2. 读用户本轮任务 envelope（任务边界）。
3. 当前事实需要时读 `.agents/briefs/current-qrp-status.md`。
4. 路由实现或审查时读 `.agents/briefs/repo-map.md`。
5. 涉及架构、命名、public surface、跨层 scope 或治理时读 `.agents/briefs/user-principles.md`。
6. 用 `references/task-modes.md` 选择 mode ID。
7. 用 `references/token-economy.md` 限制读取预算。
8. 用 `references/context-routing.md` 只加载最小相关上下文。
9. 用 `references/governance-gates.md` 选择必过 gates。
10. 涉及 public API/CLI、artifacts、data、report、private safety 或 verification semantics 时读 `references/contracts-and-verification.md`。
11. 需要实现或路由 P2/P3 时读 `references/owner-layer-routing.md`。
12. 涉及 MCP、plugins、hooks、subagents、Codex exec、GitHub Action 时读 `references/external-tools-and-adapters.md`。

## 默认执行设置

- 输出语言：面向用户的最终回复和治理 Markdown 默认中文；路径、命令、API route、字段名、错误码、status token、mode ID 保留英文。
- 分支：实现、架构、docs/rules、治理修改默认使用 `feature/<task-name>`；当前治理第二轮修补应继续在 `feature/agent-governance-redesign` 上追加 commit。
- 上下文：先 targeted `rg` 和 diff summary，再窄读文件。不要默认 broad-load private data、generated outputs、旧 handoff packages、全部 docs 或全部 templates。
- Subagents：默认不用。只有 review/goal/release/governance 等复杂并行审查可用，且默认 read-only；main agent 负责 integration、truthfulness、commit、push、handoff。
- Hooks/MCP/plugins/repo-local `.codex`：默认 draft-only，不在 repo 内直接启用。

## 任务模式

编辑前先分类：

```text
small-fix | standard-implementation | goal-closure | independent-review |
handoff-packaging | merge-gate | release-gate | docs-status-sync |
governance-redesign | apply-only-validation | ambiguous
```

如果分类不清，只有出现 blocking ambiguity（阻塞性歧义）时才问用户；不要为了小不确定性浪费一轮对话。不要把 review、handoff、merge、packaging、governance 任务改成业务实现。

## 必过 gates

中等及以上任务按 `references/governance-gates.md` 选择相关 gates：

```text
Task Router | Scope | Context | Owner Layer | Blast Radius | Red Test |
Same Issue Family Scan | Artifact Truth | Public Surface | Private Safety |
Memory/Performance Awareness | Docs/Status Sync | .run Tooling |
Script Portability | Handoff | Final Truthfulness
```

## 真实性规则

- `PASS`：检查实际运行且通过。
- `FAIL`：检查实际运行且失败；不得重命名。
- `NOT_RUN`：没有运行证据。
- `BLOCKED`：环境、依赖、权限或输入缺失导致无法运行。
- final report 必须列出实际运行、未运行和剩余风险；不要粘贴长日志，除非失败定位需要。

## 私有输出规则

任何可能持久化 private-derived row-level output（私有派生行级输出）的命令、API handler、workflow、report writer、exporter、package builder、metadata writer 或 handoff tool，都必须通过 Private Safety Gate。Public-safe bundle 必须排除 private data 和 blocked file types。

## 验证脚本

相关时使用 `scripts/` 里的跨平台 Python 脚本：

```text
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --help
python .agents/skills/quant-research-platform-maintenance/scripts/package_safety_check.py --help
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --help
```

这些脚本是 deterministic helpers（确定性辅助检查），不能替代 tests、代码审查或任务专项验证。
