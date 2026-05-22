# Handoff Before Directory Rename / 目录改名前交接

本文是改名前和新对话启动时的最短交接入口。它不替代 [PROJECT_STATUS.md](../../PROJECT_STATUS.md)、[architecture-refactor-design.md](architecture-refactor-design.md) 或 [phase-1-implementation-checklist.md](phase-1-implementation-checklist.md)。

## 当前状态

- 项目身份已经收敛为 `quant-research-platform`。
- 当前本地 checkout 仍可能位于旧目录名下；目录改名由用户手动执行。
- A 股是当前第一版完整 market module。
- 当前阶段是 Phase 1 代码重构前的文档和边界收口，不是代码重构执行中。

## 改名前建议

1. 先提交当前文档整理，避免目录改名和架构文档变更混在同一个提交里。
2. 用户手动把本地目录改成 `quant-research-platform`。
3. 新对话从项目新目录启动后，先读本文件和下方入口文档。
4. 不要在旧目录名和新目录名之间复制私有数据、生成产物或凭据。

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

- 代码重构从 [architecture-refactor-design.md](architecture-refactor-design.md) 和 [phase-1-implementation-checklist.md](phase-1-implementation-checklist.md) 开始。
- Phase 1 只处理本地研究平台重构：evaluations、artifacts、API、本地 workbench、A 股 pipeline runner 和 Factor Factory 第一版。
- 当前不要做 Phase 2、网站化、数据库、认证、外部行情 API、券商 API、paper trading、live trading、部署服务或真实交易连接。
- 长期能力如 screening、fundamental analysis、ranking、watchlists、alerts、portfolio analytics、research cards 已写入长期蓝图，但不属于当前 Phase 1 执行范围。
