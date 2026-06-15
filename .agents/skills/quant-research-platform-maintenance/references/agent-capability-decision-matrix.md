# 智能体能力取舍矩阵（Agent capability decision matrix）

本表用于判断 Codex / GPT / Claude / 其他 agents 的能力应落在何处。默认原则：先用低风险、repo-neutral（仓库中立）机制；高权限机制只保留 draft-only。

| 能力 | 当前采用 | 原因 | token / 安全成本 | 落点 |
|---|---|---|---|---|
| `AGENTS.md` | 现在使用 | Codex 会在任务前读取，适合硬边界 | 过长会浪费上下文 | 根目录短文件 |
| `.agents/skills` | 现在使用 | progressive disclosure（渐进式加载），适合任务流程 | description 过宽会误触发 | `.agents/skills/...` |
| `references/` | 现在使用 | 详细规则按需加载 | 文件太多会迷路 | 按主题拆分 |
| `templates/` | 现在使用 | 一种 mode 一个模板 | 过多会冲突 | 只读一个 primary template |
| Python guard scripts（Python 防护脚本） | 现在使用 | 机械检查 private/path/package/truth | 覆盖有限 | `scripts/*.py` |
| `.run` tools/evidence（本地工具/证据） | 按任务使用 | handoff/验证证据 | raw logs 大、private 风险 | summary-first |
| nested `AGENTS.md` | 以后再说 | 子目录规则可更精确 | 指令叠加复杂 | 暂不新增 |
| hooks | 仅草案 | 可自动 gate | trust、误拦截、跨平台 | adapter docs |
| MCP | 仅草案 / 外部工具 | Context7/GitHub 可辅助 | 网络、权限、token | 不写 repo config |
| plugins | 以后再说 | 跨 repo 分发 skills | 安装/权限/维护 | 不创建 QRP plugin |
| subagents | 受控使用 | 并行审查复杂 issue family | 高 token | prompt-level policy |
| custom subagent config（自定义子智能体配置） | 仅草案 | 可定制 reviewer | 需要 repo-local `.codex` 或 global config | `.example` only if later approved |
| `codex exec` | 仅草案 | 可做重复审查/CI | auth/环境风险 | docs only |
| GitHub Action | 以后再说 | PR/release 自动审查 | secrets/runner 权限 | future |
| automations | 以后再说 | 定期 stale check | 无人值守风险 | future |
