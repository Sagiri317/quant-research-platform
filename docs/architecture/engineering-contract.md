# Engineering Contract

本文件描述工程层当前合同，不记录当前状态。

## Python 与依赖

项目合同以 `pyproject.toml` 和 `environment.yml` 为准。文档不写用户本地解释器路径或个人环境路径。缺依赖时，应在允许范围内安装依赖并记录；无法安装写 `BLOCKED`。

## 运行与验证

常用命令：

```bash
git diff --check
python tests/tools/run_profile.py --profile fast --output-dir <repo-relative-output-dir> --tmp-dir <repo-relative-tmp-dir>
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
```

## 状态 token

只使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。超时属于 `BLOCKED`。不要写 “looks good”“probably pass” 这类不可验证语义。

## 文件路径

公开文档和 evidence 使用 repo-relative path。用户本地绝对路径、临时私有目录和供应商私有路径留在 LOCAL_ONLY 边界内。

## 代码修改原则

- 最小修改。
- 同步测试。
- 被替换实现优先收敛，不并行保留。
- 不新增 public surface，除非用户批准。
- artifact 字段过渡期需要用户批准。

## 测试职责

普通 tests 保护当前合同。owner-layer negative tests 覆盖边界回归。

测试分层由 `pyproject.toml` 的 pytest markers 和
`tests/tools/run_profile.py` 的 profile 定义共同约束。profile 清单和选择逻辑
以 runner 输出为准，文档不复制完整 pytest expression。

`slow` 只表示运行慢，`requires_local_data` 只表示需要本地数据或本地环境；
两者彼此正交。性能优化依据实际耗时画像、slowest tests/files 和
before/after 对比，不写硬编码秒数阈值，不新增独立性能门禁文件。

正式 profile runner 位于 `tests/tools/run_profile.py`。执行示例：
`python tests/tools/run_profile.py --profile fast --output-dir <repo-relative-output-dir> --tmp-dir <repo-relative-tmp-dir>`。

`.run/` 只保存输出、profile、临时文件和 handoff/review artifacts，不作为
tracked tool source。普通 `python -m pytest` 默认排除 `slow` 和
`requires_local_data`；完整收口使用 runner 的 `all` profile 显式选择。

## 治理规则事实源

`.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 是内部治理规则事实源。`verify_governance.py`、handoff source-tree scan、handoff verifier safety scan、single handoff verifier safety scan 和相关 tests 只从这里读取或派生 current-contract-only 规则、版本面规则、negative-test allowlist、source-tree scan allowlist、package safety path parts 和 status token。

边界回归由 owner-layer negative tests 和治理 gate 记录为 `FAIL`，不依赖文案关键词。

## 运行结果与 JSON

`<run_root>/run.json` 是运行级机器事实源，`<run_root>/summary.md` 是运行级人读摘要，实例结果位于 `<run_root>/outputs/<output_id>/`。QRP-owned JSON 必须是顶层 object，必须有 `schema_id`，版本不编码进 `schema_id`。

compare 按 `output_id` 配对，并使用 `output_definition_id` 与 metadata 做类型和语义校验。状态聚合只由 `policy/check_status.py` 定义。

REVIEW_SAFE evidence tests 分为纯 synthetic fast gate 和 data/parquet contract gate：

```bash
python tests/tools/run_profile.py --list-profiles
python tests/tools/run_profile.py --profile <profile> --output-dir <repo-relative-output-dir> --tmp-dir <repo-relative-tmp-dir>
```

handoff tests 的具体执行入口统一使用 `tests/tools/run_profile.py`。
