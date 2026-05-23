# Directory Rename Handoff Record / 目录迁移历史记录

本文是目录迁移前的历史交接记录。当前 checkout 已位于 `quant-research-platform`；新会话应优先阅读 [PROJECT_STATUS.md](../../PROJECT_STATUS.md)、[README.md](README.md)、[architecture-refactor-design.md](architecture-refactor-design.md) 和 [phase-1-implementation-checklist.md](phase-1-implementation-checklist.md)。

## 当前状态

- 项目身份已经收敛为 `quant-research-platform`。
- 当前本地 checkout 已位于 `quant-research-platform`。
- A 股是当前第一版完整 market module。
- 当前阶段是 Phase 1 代码重构收口和验证中；目录改名不再是当前待办。

## 历史建议

以下内容是目录迁移前的历史建议，保留用于解释上下文：

1. 先提交 Phase 1 代码、测试和文档收口，避免目录改名和架构重构混在同一个提交里。
2. 用户手动把本地目录改成 `quant-research-platform`。
3. 不要在旧目录名和新目录名之间复制私有数据、生成产物或凭据。

## 新对话先读

按顺序阅读：

1. [AGENTS.md](../../AGENTS.md)：项目安全、环境、私有数据、交易和删除边界。
2. [PROJECT_STATUS.md](../../PROJECT_STATUS.md)：当前事实、下一步和 root handoff 状态。
3. [README.md](README.md)：架构文档索引。
4. [architecture-refactor-design.md](architecture-refactor-design.md)：Phase 1 重构设计。
5. [phase-1-implementation-checklist.md](phase-1-implementation-checklist.md)：Phase 1 执行清单。

## 新对话启动语句

```text
继续 quant-research-platform，先读 AGENTS.md、PROJECT_STATUS.md、docs/architecture/handoff-before-directory-rename.md、docs/architecture/README.md、docs/architecture/architecture-refactor-design.md、docs/architecture/phase-1-implementation-checklist.md。准备执行 Phase 1 重构，不要做 Phase 2、网站化、实盘、外部 API 或目录改名。
```

## 下一步边界

- Phase 1 后续检查从 [architecture-refactor-design.md](architecture-refactor-design.md) 和 [phase-1-implementation-checklist.md](phase-1-implementation-checklist.md) 开始。
- Phase 1 只处理本地研究平台重构：evaluations、artifacts、API、本地 workbench、A 股 pipeline runner 和 Factor Factory 第一版。
- 已批准的 Phase 1 目录目标是：`interfaces/api/` 承载本地 FastAPI 适配，`interfaces/cli/` 承载 CLI，`datasets/` 承载通用 CSV / Parquet reader，`markets/implementations/a_share/` 承载 A 股市场实现，`pipelines/a_share/` 承载 A 股 workflow runner；不创建 Phase 2 的 `apps/api`，不保留空 `markets/crypto/` 实现。
- 当前不要做 Phase 2、网站化、数据库、认证、外部行情 API、券商 API、paper trading、live trading、部署服务或真实交易连接。
- 长期能力如 screening、fundamental analysis、ranking、watchlists、alerts、portfolio analytics、research cards 已写入长期蓝图，但不属于当前 Phase 1 执行范围。
