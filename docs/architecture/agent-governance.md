# QRP 智能体治理架构

本文是长期治理说明，不是日常任务 prompt。日常任务应先读 `AGENTS.md`，再按 `.agents/skills/quant-research-platform-maintenance/SKILL.md` 进行 task routing（任务路由）。

## 1. 目标

QRP 的 agent governance（智能体治理）目标是：更强约束、更少上下文、更少返工、更清晰的任务路由、更严格的 public-safe/private-safe 边界、更真实的 handoff/evidence 报告。

治理层服务以下原则：Semantic closure（语义收口）、Naming first-class（命名一等）、Replace over parallel（替代优先）、Minimal public API（最小公共 API）、Evidence over claims（证据优先）、Private-data safety（私有数据安全）、Layer-local change（局部改动）、No handoff-driven architecture（handoff 不反向驱动架构）、Extensibility without abstraction creep（可扩展但不抽象膨胀）。

## 2. 分层职责

| 层 | 读取时机 | 职责 | token 预算 |
|---|---|---|---|
| `AGENTS.md` | 每次任务自动/优先 | hard boundaries、stop conditions、task router gate、path authority | 小，稳定，目标 < 12KB |
| `.agents/briefs/current-qrp-status.md` | 需要当前 QRP 事实时 | 短 current truth，不存 Git log 和 run 细节 | 小 |
| `.agents/briefs/user-principles.md` | 架构/命名/治理/public surface/cross-layer work | 用户长期偏好 | 小 |
| `SKILL.md` | QRP maintenance/governance 任务触发时 | operational workflow、加载顺序、truth rules | 小 |
| `.agents/skills/**/references/` | 按任务类型加载 | task modes、context routing、gates、owner routing、contracts | 中，按需 |
| `.agents/skills/**/templates/` | short prompt 或特定 mode | execution checklist，不是大百科 | 小，单个加载 |
| `.agents/skills/**/scripts/` | 机械 gate | cross-platform deterministic checks | 低 context，靠执行 |
| `docs/architecture/` | 架构研究、审查、治理重构 | 长期设计与诊断 | 不日常加载 |
| `.run/` | handoff/package/evidence | ignored local evidence/tooling layer | 不作为架构源 |
| hooks/subagents/MCP/plugins | optional/draft | 只有重复且高价值时才启用 | 默认 0 |

## 3. 三层路径边界

1. global `.codex/`：用户机器级 Codex 配置/日志/历史证据源。只读；不修改；不复制 raw dump。
2. repo `.agents/`：QRP 主治理根。agent-neutral，服务 Codex/GPT/Claude/other agents。
3. repo-local `.codex/`：默认不创建。只有 Codex-specific 功能确有必要时，作为 optional/draft adapter 讨论。

同理，项目默认不新增 `.claude/` 或 `CLAUDE.md`。Claude 可以阅读通用 repo docs 和 `.agents` 内容；Claude-specific 配置应先保持 draft-only。

## 4. 为什么这样减少返工

- Task Router Gate 避免把 handoff/package/review/merge/governance 误执行成业务代码实现。
- Context Gate 让 agent 先读关键事实和 owner 文件，避免读大量无关历史或漏读核心合同。
- Owner Layer Gate 把 P2/P3 路由到归属层，避免跨层顺手修。
- Artifact Truth Gate 把 `PASS/FAIL/NOT_RUN` 作为合同，避免 final report 虚假完成。
- Private Safety Gate 和 package safety script 把 handoff 安全从人工记忆变成机械 gate。
- Script Portability Gate 消除 ps1-only 对 Win11/macOS/Linux/WSL 的不兼容。
- docs/status sync 只同步 active docs，避免旧 N1/N2/N3 叙事污染当前 truth。

## 5. 默认不启用的能力

- hooks：只有当必须“每次都机械执行”且脚本已跨平台验证时才启用。当前保持 draft-only。
- MCP：只有当上下文在 repo 外且频繁变化、且有明确安全边界时才启用。当前保持 draft-only。
- subagents：只在复杂审查/goal closure 中按 issue family 使用，默认 read-only。
- plugins：仅当需要跨 repo 分发 skills/MCP/hooks 时研究；当前不启用。
- repo-local `.codex/`：只作为 draft adapter，不作为主治理根。

## 6. 维护规则

治理文件本身也要遵守 replace-over-parallel：旧模板如果被新 mode matrix 替代，应删除而不是保留 alias。新增规则前先确认是否可以放入现有 gate、template 或 script；不要把 `AGENTS.md` 扩写成长篇百科。
