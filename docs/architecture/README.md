# Architecture Documents / 架构文档

Phase 1 已实现代码仍需按 `architecture-refactor-design.md` 和 `phase-1-implementation-checklist.md` 复核；这两个文件是验收基准，本索引只记录阅读顺序和阶段边界。当前已批准的结构决策是：本地 API -> `interfaces/api/`，CLI -> `interfaces/cli/`，通用 reader -> `datasets/`，A 股市场实现 -> `markets/implementations/a_share/`，A 股 workflow runner -> `pipelines/a_share/`；Phase 1 不创建 `apps/api`，不保留空 `markets/crypto` 实现。

本目录收拢 `quant-research-platform` 的架构蓝图、能力池和工程治理规则。

阅读顺序：

1. [target-directory-structure.md](target-directory-structure.md)：这些能力最终放在哪里，以及 Phase 1 / Phase 2 / Future 的阶段边界。
2. [engineering-standards.md](engineering-standards.md)：模块依赖、合同、版本、错误、配置、测试、扩展和生成物规则。
3. [quant-research-standards.md](quant-research-standards.md)：point-in-time、universe、tradability、成本、容量、统计验证和复现规则。
4. [architecture-refactor-design.md](architecture-refactor-design.md)：Phase 1 contract-first 架构重构设计和验收依据。
5. [phase-1-implementation-checklist.md](phase-1-implementation-checklist.md)：把 Phase 1 重构设计拆成可执行任务、验收项和测试矩阵。
6. [future-capabilities.md](future-capabilities.md)：未来要做什么，以及哪些能力不属于当前已实现范围。
7. [handoff-before-directory-rename.md](handoff-before-directory-rename.md)：历史目录迁移交接记录，仅作背景参考。

本目录中的文档是设计和治理依据，不代表所有目录或能力都已经实现。
