# 治理验证清单（Governance verification checklist）

治理层变更后，至少检查以下项目。未运行必须写 `NOT_RUN` 或 `BLOCKED`。

## 路径与范围

- 只改治理层：`AGENTS.md`、`README.md`、`PROJECT_STATUS.md`、`.agents/**`、治理相关 `docs/architecture/**`。
- 不改 `src/**`、业务 `tests/**`、`pyproject.toml`、`environment.yml`。
- 不创建 repo-local `.codex/`、`.claude/`、`CLAUDE.md`、hooks/MCP/plugin runtime config。
- 不加入 private data、blocked extensions、旧 handoff zip/json。

## 内容一致性

- 人读文件中文为主，路径/命令/API/code symbol/mode ID/status token 保留英文。
- `AGENTS.md` 足够短且包含 hard boundaries、task router、stop conditions。
- task mode ID 在 `AGENTS.md`、`SKILL.md`、`task-modes.md`、templates 中一致。
- current status 不保留旧 branch/next step。
- 内存预算等具体数字以代码事实为准。
- templates 支持短提示词精准路由，不要求用户每次写长 prompt。

## 建议命令

```text
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --repo-root . --governance-only
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python -c "from pathlib import Path; import py_compile; [py_compile.compile(str(p), doraise=True) for p in Path('.agents/skills/quant-research-platform-maintenance/scripts').glob('*.py')]"
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/agents tests/architecture -q
```

Windows PowerShell 下设置 pytest 环境变量：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
python -m pytest tests/agents tests/architecture -q
Remove-Item Env:\PYTEST_DISABLE_PLUGIN_AUTOLOAD
```
