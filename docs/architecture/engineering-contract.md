# Engineering Contract

本文件描述工程层当前合同，不记录当前状态。

## Python 与依赖

项目合同以 `pyproject.toml` 和 `environment.yml` 为准。文档不写用户本地解释器路径、个人环境路径或固定 conda 环境名。本地依赖、构建、测试和验证只使用 conda；环境名从 `environment.yml` 动态读取。无法满足环境时写 `BLOCKED`。

如果 `pyproject.toml` dev dependencies 包含 `build`，`environment.yml` 必须包含对应 conda 包 `python-build`。`pyproject.toml` 必须显式声明 `[build-system]`，且 `environment.yml` 必须显式提供对应构建后端依赖。package smoke 使用 conda-managed non-isolated build：`python -m build --no-isolation --sdist --wheel --outdir <dist>`；package smoke 不执行安装，只允许 wheel 解压后通过 `PYTHONPATH` import smoke。

## 运行与验证

常用命令：

```bash
git diff --check
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py --repo-root . --json
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
```

## 状态 token

只使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。超时属于 `BLOCKED`。不要写 “looks good”“probably pass” 这类不可验证语义。

每个 `PASS` 只覆盖实际运行的 command、scope 和 artifact。profile runner、governance verifier、handoff verifier、source-tree scan、package safety scan 和 reviewer evidence 不能互相替代。top-level summary 不得掩盖子项的 `FAIL`、`NOT_RUN` 或 `BLOCKED`。

## 文件路径

公开文档和 evidence 使用 repo-relative path。用户本地绝对路径、临时私有目录和供应商私有路径留在 LOCAL_ONLY 边界内。

package evidence 描述 handoff package 的成员、hash、manifest、sidecar 和 Git bundle。reviewer evidence 描述 review role、scope、files inspected、commands run、findings、fixes 和 limitations。二者必须分开记录；handoff layout、sidecar、task metadata、validation summary、`FieldSpec` 和 `allowed_keys()` 合同由 `.agents/skills/quant-research-platform-maintenance/scripts/handoff/handoff_contract.py` 拥有，receipt schema specs、invariants、normalization 和 comparison 由 `.agents/skills/quant-research-platform-maintenance/scripts/handoff/receipt_contract.py` 拥有，治理扫描规则由 `.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 拥有。

handoff 使用 compact replayable receipts：machine validation rows 只来自 replay-backed receipts；verifier result path 只做一次 zip inspection，并在需要 replay 时只 clone `final-branch.bundle` 一次，在同一个 clone 中完成 share-safety scan、receipt replay 和 count/hash comparison。Codex 不应手写 handoff evidence，也不应把 raw grep output、full pytest log、完整 stdout/stderr、wheel、sdist、完整 review transcript 或本地 `.run/local/**` 产物打入 handoff。

正式 handoff success 还要求 mandatory validation rows 全部为 `PASS`。`validation-summary.json` 的顶层 status 只由 rows 聚合派生；任一 mandatory row 为 `FAIL`、`NOT_RUN`、`BLOCKED`、缺失、重复或未知状态时，builder 和 verifier 都必须返回失败。

## 代码修改原则

- 最小修改。
- 同步测试。
- 被替换实现优先收敛，不并行保留。
- 不新增 public surface，除非用户批准。
- artifact 字段过渡期需要用户批准。
- owner-local internal API、private module pattern 和 storage/artifacts 边界由 references 与 executable gates 执行，本文不复制细则。

## 测试职责

普通 tests 保护当前合同。owner-layer negative tests 覆盖边界回归。

测试分层由 `pyproject.toml` 的 pytest markers 和
`tests/tools/run_profile.py` 的 profile 定义共同约束。profile 清单和选择逻辑
以 runner 输出为准，文档不复制完整 pytest expression。

`integration`、`mutation`、`slow`、`performance` 和
`requires_local_data` 必须由对应 profile 或 `all` profile 显式选择。
`slow` 只表示运行慢，`requires_local_data` 只表示需要本地数据或本地环境；
两者彼此正交。`performance` 是预留 profile；没有 performance tests 时
0 collected 记为 `NOT_RUN`，不能记为 `PASS`。性能优化依据实际耗时画像、
slowest tests/files 和 before/after 对比，不写硬编码秒数阈值，不新增独立性能门禁文件。

正式 profile runner 位于 `tests/tools/run_profile.py`。执行示例：
`conda run -n <environment.yml name> python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast`。
`profile_smoke` handoff validation row 只来自 runner 的 `profile-smoke` profile。

`.run/` 只保存输出、profile、临时文件和 handoff/review artifacts，不作为
tracked tool source。普通 `python -m pytest` 是开发者默认非重型 suite，
默认排除 `integration`、`mutation`、`slow`、`performance` 和
`requires_local_data`；正式 evidence 使用 `tests/tools/run_profile.py`，
完整收口使用 runner 的 `all` profile 显式选择。

## 治理规则事实源

`.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 是内部治理规则事实源。`verify_governance.py`、source-tree scan、bundle share-safety scan、共享安全路径规则和相关 tests 只从这里读取或派生 current-contract-only 规则、版本面规则、negative-test allowlist、source-tree scan allowlist、package safety path parts 和 status token。

边界回归由 owner-layer negative tests 和治理 gate 记录为 `FAIL`，不依赖文案关键词。

## 运行结果与 JSON

`<run_root>/run.json` 是运行级机器事实源，`<run_root>/summary.md` 是运行级人读摘要，实例结果位于 `<run_root>/outputs/<output_id>/`。QRP-owned JSON 必须是顶层 object，必须有 `schema_id`，版本不编码进 `schema_id`。

compare 按 `output_id` 配对，并使用 `output_definition_id` 与 metadata 做类型和语义校验。状态聚合只由 `policy/check_status.py` 定义。

REVIEW_SAFE evidence tests 分为纯 synthetic fast gate 和 data/parquet contract gate：

```bash
conda run -n <environment.yml name> python tests/tools/run_profile.py --list-profiles
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile <profile> --run-dir .run/<purpose>/<profile>
```

handoff tests 的具体执行入口统一使用 `tests/tools/run_profile.py`。测试位置按 owner layer 放置；详细 placement 规则由 maintenance references 和 architecture gates 拥有。

handoff builder/verifier 位于 `.agents/skills/quant-research-platform-maintenance/scripts/handoff/`。正式生成入口是 `build_handoff_package.py --repo-root . --out-dir .run/handoffs --task-name <task-name> --base-ref <base-ref>`；builder 只读取当前 Git state 和这些 CLI options，并通过 replay 自动生成 machine evidence，然后永远通过 `verify_handoff_package_result()` 运行 verifier。`verify_handoff_package_result()` 是唯一 package-level verifier function；standalone verifier CLI 保留为 verifier 的直接入口，未经用户明确批准不新增 wrapper、alias、adapter 或 compatibility layer。builder 不运行独立 safety scan；share-safety status、handoff verifier status 和 zip sha256 来自同一个 verification result。Verifier 从 `final-branch.bundle` clone replay 并比较 packaged receipts、validation summary 和 manifest。具体 layout、schema、receipt 和 active denied semantics 由 executable contracts、verifier 和 tests 拥有。
