# 文档状态同步（Docs/status sync）

用于 active docs、status、brief、code-map 的同步。默认 docs-only，不改业务代码。

## 同步范围

常见文件：

```text
README.md
PROJECT_STATUS.md
AGENTS.md
.agents/briefs/current-qrp-status.md
.agents/briefs/repo-map.md
.agents/briefs/user-principles.md
docs/architecture/README.md
docs/architecture/open-issues-register.md
```

不要默认翻旧 archive、旧 handoff、全量 docs。

## 检查点

- 分支和 next step 是否过期。
- public API/CLI 是否和代码一致。
- private roots 是否一致。
- Python / memory budget 等数字是否以代码为事实源。
- 人读文件是否中文为主。
- 旧 N1/N2/N3 status 是否误保留为当前事实。

## 验证

```text
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --repo-root . --governance-only
git diff --check
```
