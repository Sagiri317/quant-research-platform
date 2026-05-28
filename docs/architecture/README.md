# Architecture Documents / 架构文档

Phase 1 已完成代码功能复核和文档收口；验收结论已沉淀到 [phase-1-acceptance-record.md](phase-1-acceptance-record.md)。2026-05-27 已在 N8 合入 `main` 后同步 `docs/code-map/` 与当前实现状态，并把历史设计、清单和目录迁移交接内容合并为当前文档集。已批准并落地的结构决策是：本地 API -> `interfaces/api/`，CLI -> `interfaces/cli/`，通用 reader -> `datasets`，A 股市场实现 -> `markets/implementations/a_share`，A 股 workflow runner -> `pipelines/a_share`；Phase 1 不创建 `apps/api`，不保留空 `markets/crypto` 实现。

本目录收拢 `quant-research-platform` 的架构蓝图、能力池和工程治理规则。

阅读顺序：

1. [target-architecture-and-capabilities.md](target-architecture-and-capabilities.md)：最终功能和架构目标；合并原目标目录结构和未来能力池，保留目标细节，同时标明当前已落地边界。
2. [ai-native-engineering-principles.md](ai-native-engineering-principles.md)：AI-native engineering mental model 和长期工程原则；不是当前实现状态、roadmap 或 Codex 执行规则。
3. [engineering-standards.md](engineering-standards.md)：模块依赖、合同、版本、错误、配置、测试、扩展和生成物规则。
4. [quant-research-standards.md](quant-research-standards.md)：point-in-time、universe、tradability、成本、容量、统计验证和复现规则。
5. [phase-1-acceptance-record.md](phase-1-acceptance-record.md)：Phase 1 已实现功能、验收证据、冻结合同和遗留架构债。
6. [next-stage-roadmap.md](next-stage-roadmap.md)：Phase 1 之后的下一阶段任务、验收口径和测试边界。
7. [next-stage-worktree-plan.md](next-stage-worktree-plan.md)：下一阶段已完成和后续分支、工作树、依赖和最低验收拆分。

本目录中的文档是设计和治理依据，不代表所有目录或能力都已经实现。
