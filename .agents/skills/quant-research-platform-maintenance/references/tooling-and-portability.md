# Tooling and Portability

本文件定义治理脚本、外部工具和可移植性边界。

## 1. 脚本原则

- 脚本应可在 repo root 运行。
- 路径使用相对 repo root 的 POSIX 风格。
- 不写死用户本地 Python 路径、conda 环境名、Windows 盘符或私有数据路径。
- 输出应可被 human 和 machine 读取。
- 检查失败时返回非零 exit code。

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

## 4. Subagent 和 review-fix loop

可以用 subagent 做只读扫描或分工 review，但最终责任在主 agent。subagent 结果必须被主 agent 复核，不能原样当事实。结束前必须汇总并关闭 subagent。

review-fix loop 应小步执行：scan → classify → patch → validate → rescan。不要在未确认 scope 前做大范围重构。

## 5. 禁止配置扩张

未经用户批准，禁止新增 repo-local `.codex/`、global `.codex/`、hooks、MCP config、plugin runtime config、custom subagent runtime config。文档可以说明未来方向，但当前任务默认不启用这些机制。

`.agents/` 只承载治理、handoff 和 prompt tooling，不得承载 runtime 业务逻辑。
