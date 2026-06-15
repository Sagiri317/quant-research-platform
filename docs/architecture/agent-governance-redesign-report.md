# 智能体治理重构报告（Agent governance redesign report）

本文记录本轮治理体系的诊断、目标架构、功能取舍、token economy、迁移计划和验证要求。它是 architecture reference（架构参考），不是日常任务 prompt。

## 1. 诊断

主要返工来源：任务模式混淆、public surface 无意扩张、artifact truthfulness 风险、private safety 风险、`.codex/.agents/.run` 路径权威混淆、PowerShell-only 脚本、active status drift、上下文读取过多。

## 2. 目标

治理层必须满足：

```text
短提示词精准路由
+ 用户原则自动生效
+ 一遍过 gates
+ 少 token
+ 少返工
+ public-safe/private-safe 边界清楚
+ handoff/evidence 真实
```

## 3. 目标架构

| 层 | 作用 | 为什么省 token |
|---|---|---|
| `AGENTS.md` | 每次任务都读的短硬边界 | 不把百科塞进默认上下文 |
| `.agents/briefs` | 当前事实和长期偏好 | 不翻旧 handoff 或长对话 |
| `SKILL.md` | QRP 操作路由 | 统一加载顺序 |
| `references` | 按主题拆分细则 | 只在需要时读 |
| `templates` | 每种 mode 一个执行模板 | 一次只读一个 |
| `scripts` | 机械检查 | 少让模型读日志/规则 |
| `docs/architecture` | 长期解释 | 日常不读 |
| `.run` | 本地证据工具 | summary-first，不读 raw logs |

## 4. 能力取舍

- AGENTS.md：现在使用，保持短。
- `.agents/skills`：现在使用，依赖 progressive disclosure（渐进式披露）。
- hooks：draft-only。
- MCP：外部可用，repo 不配置。
- plugins：外部可用，QRP 不创建 plugin。
- subagents：prompt-level controlled use，默认 read-only review。
- `codex exec` / GitHub Action / automations：future/draft-only。

## 5. Token economy

硬规则见 `references/token-economy.md`。核心是：diff-first、file-second、repo-last；一次只读一个模板；`.run` summary-first；`rg` 命中摘要先于全文；final report 不粘贴长日志；外部工具和 subagents 都有 token gate。

## 6. 验证要求

治理层变更至少运行：

```text
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --repo-root . --governance-only
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python -c "from pathlib import Path; import py_compile; [py_compile.compile(str(p), doraise=True) for p in Path('.agents/skills/quant-research-platform-maintenance/scripts').glob('*.py')]"
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/agents tests/architecture -q
```

未运行检查必须写 `NOT_RUN` 或 `BLOCKED`。

## 7. 迁移计划

- Phase 1：中文化人读文件、修正代码事实冲突、统一 mode ID、删除一次性 apply brief、补 token economy 和外部工具共存规则。
- Phase 2：在用户 Win11 + Python 3.13.14 环境验证脚本；post-merge status sync 通过后视为完成。
- Phase 3：必要时把稳定脚本转成 draft hooks / `codex exec` 示例；仍不默认启用 repo-local `.codex/`。
