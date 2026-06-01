# Architecture Documents / 架构文档

Phase 1 已完成代码功能复核和文档收口；验收结论已沉淀到 [phase-1-acceptance-record.md](phase-1-acceptance-record.md)。2026-05-27 已在 N8 合入 `main` 后同步 `docs/code-map/` 与当前实现状态，并把历史设计、清单和目录迁移交接内容合并为当前文档集。已批准并落地的结构决策是：本地 API -> `interfaces/api/`，CLI -> `interfaces/cli/`，通用 reader -> `datasets`，A 股市场实现 -> `markets/implementations/a_share`，A 股 workflow runner -> `pipelines/a_share`；Phase 1 不创建 `apps/api`，不保留空 `markets/crypto` 实现。

本目录收拢 `quant-research-platform` 的架构蓝图、能力池和工程治理规则。

Human reading order:

1. [target-architecture-and-capabilities.md](target-architecture-and-capabilities.md)：最终功能和架构目标；合并原目标目录结构和未来能力池，保留目标细节，同时标明当前已落地边界。
2. [ai-native-engineering-principles.md](ai-native-engineering-principles.md)：AI-native engineering short mental model / compressed principles；不是当前实现状态、roadmap 或 Codex 执行规则。
3. [ai-native-engineering-standard.md](ai-native-engineering-standard.md)：AI-native engineering full governance reference / complete standard；低频阅读，普通任务不要默认加载全文，优先用 `rg` 定位小片段。
4. [engineering-standards.md](engineering-standards.md)：模块依赖、合同、版本、错误、配置、测试、扩展和生成物规则。
5. [quant-research-standards.md](quant-research-standards.md)：point-in-time、universe、tradability、成本、容量、统计验证和复现规则。
6. [phase-1-acceptance-record.md](phase-1-acceptance-record.md)：Phase 1 已实现功能、验收证据、冻结合同和遗留架构债。
7. [next-stage-roadmap.md](next-stage-roadmap.md)：Phase 1 之后的下一阶段任务、验收口径和测试边界。

Codex execution templates are not maintained in architecture docs. Current
task-mode, branch, commit, push, fast-forward merge, and Goal / Overnight
execution rules live in `.agents/skills/quant-research-platform-maintenance/references/templates/`, the project maintenance skill, and
its context-routing reference.

Codex context loading order:

1. 普通实现或修复任务先读根目录 [../../AGENTS.md](../../AGENTS.md)，再按任务表面加载相关当前文件。
2. artifact schema、`manifest.display`、API/download、frontend evidence、generated runs 任务先读 `.agents/skills/quant-research-platform-maintenance/references/contracts-and-verification.md`。
3. 诊断语义、tradability、cost、capacity、risk、benchmark 或 trading-adjacent wording 任务再读 [quant-research-standards.md](quant-research-standards.md)。
4. 模块边界、合同版本、测试、生成物、API/frontend 边界或安全规则任务再读 [engineering-standards.md](engineering-standards.md)。
5. Decision Task、architecture planning 或 future capability boundary task 才默认读取 [target-architecture-and-capabilities.md](target-architecture-and-capabilities.md)；普通 coding task 不默认加载它。
6. 普通 coding task 不默认读取 [ai-native-engineering-standard.md](ai-native-engineering-standard.md)；governance、architecture、standards、Codex workflow redesign、task-mode policy 或 documentation topology tasks 也应先用 `rg` 定位小片段，只有用户明确要求 review/rewrite full standard 或窄读无法解决治理冲突时才加载全文。
7. Branch / commit / push / merge / fast-forward merge tasks should use `.agents/skills/quant-research-platform-maintenance/references/templates/commit-and-push.md`, `.agents/skills/quant-research-platform-maintenance/references/templates/branch-push.md`, and `.agents/skills/quant-research-platform-maintenance/references/templates/ff-merge-and-push.md` instead of architecture docs.

Archive candidates:

- `docs/archive/superpowers/` 是已归档 historical execution records。引用它时应分类为 historical reference 或仍被当前文档明确继承的 active reference。
- `docs/archive/operations/next-stage-worktree-plan-2026-05.md` 是 historical operations planning record；当前 Git/worktree/task-mode execution 以 `.agents/skills/quant-research-platform-maintenance/references/templates/*`、maintenance skill 和 context-routing 为准。

本目录中的文档是设计和治理依据，不代表所有目录或能力都已经实现。
