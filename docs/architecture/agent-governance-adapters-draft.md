# Agent Governance Optional Adapters / draft-only

本文只记录 hooks、MCP、subagents、plugins、repo-local `.codex/`、Claude-specific adapters 的未来方案。当前不启用，不作为 Codex/Claude 配置文件，不要求本地信任。

## 1. Codex hooks / draft-only

可考虑的 hooks：

- `Stop` 或 final-turn hook：运行 `verify_governance.py --governance-only`，阻止治理任务误改 `src/**`。
- `PreToolUse` hook：阻止写入 private roots、`.env`、credentials、blocked package extensions。
- `UserPromptSubmit` hook：检测 task mode 是否缺失并提示先 routing。

暂不落地原因：Codex project-local hooks 需要 repo-local `.codex/` trust；本项目默认不创建 repo-local `.codex/`。hooks 还会引入信任、调试、平台兼容和误拦截成本。

测试方式：在单独 sandbox repo 中创建 `.codex/hooks.json`，用无私有数据 fixture 测试 allow/deny；失败时删除 `.codex/` 即回滚。

## 2. MCP / draft-only

可考虑 MCP：

- GitHub issue/PR MCP：用于 PR review 或 open issue 同步。
- Local docs index MCP：用于检索大型文档，而不是把全文塞入 prompt。
- Calendar/task MCP：不适用于当前 QRP 代码治理。

暂不落地原因：当前证据和 source of truth 均在 repo、input pack、`.run` 摘要中；MCP 会增加凭据、网络、权限和上下文成本。

## 3. Subagents / controlled use

可用场景：large goal closure、pre-merge review、release gate。按 issue family 分配：private safety、public surface、artifact truth、performance/memory、docs/status、script portability。默认 read-only；main agent 负责 integration 和 truthfulness。

不建议现在新增固定 subagent 配置文件。不同工具对 subagent 配置位置不同，过早落地会造成 agent-specific 维护负担。

## 4. Plugins / draft-only

可在未来把 QRP governance skill 打包成 plugin，用于多 repo 分发。当前 QRP 是单 repo，本地 `.agents` 足够。plugins 会增加 marketplace/manifest/versioning 成本，暂不落地。

## 5. Claude-specific adapters / draft-only

Claude Code 支持 `CLAUDE.md`、skills、hooks、MCP、subagents 和 plugins；但 QRP 当前不新增 `CLAUDE.md` 或 `.claude/`，避免把项目治理复制到 agent-specific 文件。若未来长期使用 Claude Code，可生成一个很短的 `CLAUDE.md`，内容只指向 `AGENTS.md` 和 `.agents`，不得复制完整规则。

## 6. 回滚原则

任何 adapter 都必须：

1. 独立分支落地。
2. 无 business code 改动。
3. 有跨平台 dry-run/check-only 验证。
4. 可通过删除 adapter 文件完整回滚。
5. 不改变 `.agents/` 作为主治理根。
