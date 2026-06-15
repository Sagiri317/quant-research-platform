# 精确应用与验证（Apply-only validation）

用于外部 patch/zip。职责是 apply exactly（精确应用）和 validate only（只验证），不是重新设计。

## 规则

1. 先确认分支和 clean working tree。
2. `git apply --check <patch>` 通过后再 apply。
3. patch 冲突时停止并报告；不要自行重写设计。
4. zip 只作 fallback/reference，除非用户明确要求按文件替换。
5. 不改 forbidden paths，不创建 repo-local `.codex/`。

## 必跑验证

```text
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --repo-root . --governance-only
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python -c "from pathlib import Path; import py_compile; [py_compile.compile(str(p), doraise=True) for p in Path('.agents/skills/quant-research-platform-maintenance/scripts').glob('*.py')]"
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/agents tests/architecture -q
```

Windows PowerShell 下用：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
python -m pytest tests/agents tests/architecture -q
Remove-Item Env:\PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

## Commit 条件

全部必要检查 `PASS` 且无 forbidden paths 后，才允许 commit/push。未运行的检查必须写 `NOT_RUN` 或 `BLOCKED`。
