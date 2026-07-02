# Tooling and Portability

本文件定义治理脚本、外部工具和可移植性边界。

## 1. 脚本原则

- 脚本应可在 repo root 运行。
- 路径使用相对 repo root 的 POSIX 风格。
- 不写死用户本地 Python 路径、conda 环境名、Windows 盘符或私有数据路径。
- 输出应可被 human 和 machine 读取。
- 检查失败时返回非零 exit code。
- Handoff builder 从 Git state 和 replay 自动生成 compact machine receipts；不要把 raw grep output、完整 pytest log、完整 stdout/stderr、wheel/sdist 或本地临时安装目录写入 handoff。
- Builder 只读取当前 Git state 和正式 CLI options；`.run/` 只保存 output/evidence，不作为脚本来源。

## 2. Python 检查

检查脚本语法时应递归扫描：

```bash
python - <<'PY'
from pathlib import Path
import py_compile
for path in Path('.agents/skills/quant-research-platform-maintenance/scripts').rglob('*.py'):
    py_compile.compile(str(path), doraise=True)
PY
```

不要只用 `glob('*.py')`，否则会漏掉 nested handoff scripts。

## 3. 外部工具

允许使用 git、rg、python、pytest 和项目已有脚本。需要安装依赖时，优先使用用户允许的环境策略。无法安装时写 `BLOCKED`。

## 4. Human review 和 review-fix loop

human/advisory review 是独立人类审查输出，不进入 machine handoff，也不作为 machine validation `PASS` 的来源。

review-fix loop 应小步执行：scan → classify → patch → validate → rescan。重验证在 review-fix loop 完成后执行，不要在未确认 scope 前做大范围重构。

人工 review 是独立的人类审查输出，不进入 machine handoff，也不作为 machine validation `PASS` 的来源；不得打包完整 transcript。

## 5. 禁止配置扩张

未经用户批准，禁止新增 repo-local `.codex/`、global `.codex/`、hooks、MCP config、plugin runtime config、custom subagent runtime config。文档可以说明未来方向，但当前任务默认不启用这些机制。

`.agents/` 只承载治理、handoff 和 prompt tooling，不得承载 runtime 业务逻辑。
