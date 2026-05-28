# Superpowers Plans and Specs

本目录保存历史 spec、implementation plan 和 agent 执行记录，用于追溯设计决策、验收背景和当时的任务拆分。

Archive boundary: 本目录是 archive candidate。当前不移动、不删除、不批量改写历史文件；如未来批准移动，优先方案是移动到 `docs/archive/superpowers/`，并同步更新当前文档中的 active references 与历史文件内的必要相对链接。

这些文件不是当前可执行入口。文件中的脚本名、验收 wrapper、长命令、subagent 分工和 cold review loop 可能已经被删除、迁移或沉淀为更通用的流程。

当前可复用维护流程以项目维护 skill 为准：

- `.codex/skills/quant-research-platform-maintenance/SKILL.md`
- `.codex/skills/quant-research-platform-maintenance/references/contracts-and-verification.md`
- `.codex/skills/quant-research-platform-maintenance/references/code-map-flow.md`

当前项目事实和用户入口以根目录文档、`docs/architecture/`、`docs/code-map/`、源码和测试为准。引用本目录内容时，应先判断它是 historical context、negative/rejection coverage、还是仍被当前合同明确继承。
