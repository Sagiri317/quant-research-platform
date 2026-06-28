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

只使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。超时属于 `BLOCKED`。不要写 “looks good”“probably pass” 这类不可验证语义。

## 文件路径

公开文档和 evidence 使用 repo-relative path。禁止写入用户本地绝对路径、临时私有目录或供应商私有路径。

## 代码修改原则

- 最小修改。
- 同步测试。
- 删除旧实现优先于并行保留。
- 不新增 public surface，除非用户批准。
- 不为 retired artifact 字段提供隐式兼容。

## 测试职责

普通 tests 保护当前合同。retired-surface negative tests 阻止禁用表面复活。

## 治理规则事实源

`.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 是内部治理规则事实源。`verify_governance.py`、handoff source-tree scan、handoff verifier safety scan、single handoff verifier safety scan 和相关 tests 只能从这里读取或派生 active docs、retired paths、retired naming、path/name/symbol/structured-key/synthesized-string patterns、version surface policy、stale truth、retired public surface、negative-test allowlist、source-tree scan allowlist、package forbidden path parts 和 status token。

Retired governance paths 不能复活；路径本身存在即 `FAIL`，不依赖文案关键词。

Retired stage naming 不能通过文件名、路径名、Python symbol、JSON/YAML/TOML key 或静态字符串拼接复活。合法版本面使用 `QRP 0.1`、`0.1`、`0.1.0`、`/api/0.1`、`definition_version` 和 `workflow_version`；QRP-owned JSON 使用不带版本后缀的 `schema_id`。带 `v` 前缀的 `0.1` API 版本路径只允许出现在规则定义或 negative tests。

## 运行结果与 JSON

`<run_root>/run.json` 是运行级机器事实源，`<run_root>/summary.md` 是运行级人读摘要，实例结果位于 `<run_root>/outputs/<output_id>/`。QRP-owned JSON 必须是顶层 object，必须有 `schema_id`，不得把版本编码进 `schema_id`。

compare 按 `output_id` 配对，并使用 `output_definition_id` 与 metadata 做类型和语义校验。状态聚合只由 `policy/check_status.py` 定义。

share-safe evidence tests 分为纯 synthetic fast gate 和 data/parquet contract gate：

```bash
python -m pytest tests/architecture/test_share_safe_evidence.py -q
python -m pytest tests/architecture/test_share_safe_evidence_data_contract.py -q -m data_dependency
```

handoff tests 分为 fast contract gate 和 slow integration gate；具体命令记录在 `README.md`、maintenance skill 和 verification reference 中。
