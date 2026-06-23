# Engineering Contract

本文件描述工程层当前合同，不记录当前状态。

## Python 与依赖

项目合同以 `pyproject.toml` 和 `environment.yml` 为准。文档不写用户本地解释器路径或个人环境路径。缺依赖时，应在允许范围内安装依赖并记录；无法安装写 `BLOCKED`。

## 运行与验证

常用命令：

```bash
git diff --check
python -m pytest tests/agents tests/architecture -q
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
```

## 状态 token

只使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`、`TIMEOUT`。不要写 “looks good”“probably pass” 这类不可验证语义。

## 文件路径

公开文档和 evidence 使用 repo-relative path。禁止写入用户本地绝对路径、临时私有目录或供应商私有路径。

## 代码修改原则

- 最小修改。
- 同步测试。
- 删除旧实现优先于并行保留。
- 不新增 public surface，除非用户批准。
- 不为旧 artifact 字段提供隐式兼容。

## 测试职责

普通 tests 保护当前合同。retired-surface negative tests 保护旧表面不复活。不要让普通 tests 保护历史结构。

## 治理规则事实源

`.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 是内部治理规则事实源。`verify_governance.py`、handoff source-tree scan、handoff safety、package safety 和相关 tests 只能从这里读取或派生 active docs、retired paths、retired naming、path/name/symbol/synthesized-string patterns、version surface policy、stale truth、retired public surface、negative-test allowlist、source-tree scan allowlist、package forbidden path parts 和 status token。

Retired governance paths 不能复活；路径本身存在即 `FAIL`，不依赖文案关键词。

Retired stage naming 不能通过文件名、路径名、Python symbol 或静态字符串拼接复活。合法版本面使用 `QRP 0.1`、`0.1`、`0.1.0`、`/api/0.1` 和 schema/contract `0.1` 字段；带 `v` 前缀的 `0.1` API 版本路径只允许出现在规则定义或 negative tests。
