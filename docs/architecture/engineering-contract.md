# Engineering Contract

本文件描述工程层当前合同，不记录当前状态。

## Python 与依赖

项目合同以 `pyproject.toml` 和 `environment.yml` 为准。文档不写用户本地解释器路径、个人环境路径或固定 conda 环境名。本地依赖、构建、测试和验证只使用 conda；环境名从 `environment.yml` 动态读取。无法满足环境时写 `BLOCKED`。

如果 `pyproject.toml` dev dependencies 包含 `build`，`environment.yml` 必须包含对应 conda 包 `python-build`。`pyproject.toml` 必须显式声明 `[build-system]`，且 `environment.yml` 必须显式提供对应构建后端依赖。package smoke 使用 conda-managed non-isolated build：`python -m build --no-isolation --sdist --wheel --outdir <dist>`；package smoke 不执行安装，只允许 wheel 解压后通过 `PYTHONPATH` import smoke。

`pytest-xdist` 只属于 dev dependencies 和 `environment.yml`，用于 `all-xdist` speed gate；具体版本以 root config 当前声明为准；它不得进入 runtime dependencies、`build-system.requires` 或 package smoke runtime。

Replay environment preflight 只检查 handoff replay/package smoke 的最小 Python runtime metadata。Python 版本和 replay 必需 distribution 约束从 `pyproject.toml` 派生；`environment.yml` 只通过 structure audit / environment contract 与其同步，不作为 runtime preflight 来源。

Structure audit 的 `tooling_environment_dependency_sync` 只检查 handoff replay、package smoke 和 speed/lint gate 需要的关键工具依赖、Python pin 与 `ruff target-version`。它不是 full dependency lockfile，不要求把全部 runtime dependencies 纳入同步 gate。

## 运行与验证

常用命令：

```bash
git diff --check
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py --repo-root . --json
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile all --run-dir .run/<purpose>/all -- --durations=100
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile all --run-dir .run/<purpose>/all-xdist -- -n 4 --dist loadfile --durations=100
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
```

## 状态 token

只使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。超时属于 `BLOCKED`。不要写 “looks good”“probably pass” 这类不可验证语义。

每个 `PASS` 只覆盖实际运行的 command、scope 和 artifact。profile runner、governance verifier、handoff verifier、source-tree scan、package safety scan 和 reviewer evidence 不能互相替代。top-level summary 不得掩盖子项的 `FAIL`、`NOT_RUN` 或 `BLOCKED`。

## 文件路径

公开文档和 evidence 使用 repo-relative path。用户本地绝对路径、临时私有目录和供应商私有路径留在 LOCAL_ONLY 边界内。

package evidence 描述 handoff package 的成员、hash、manifest、sidecar 和 Git bundle。reviewer evidence 描述 review role、scope、files inspected、commands run、findings、fixes 和 limitations。二者必须分开记录；handoff layout、sidecar、task metadata、validation summary、`FieldSpec` 和 `allowed_keys()` 合同由 `.agents/skills/quant-research-platform-maintenance/scripts/handoff/handoff_contract.py` 拥有，receipt schema specs、invariants、normalization 和 comparison 由 `.agents/skills/quant-research-platform-maintenance/scripts/handoff/receipt_contract.py` 拥有，治理扫描规则由 `.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 拥有。

handoff 使用 compact replayable receipts：machine validation rows 只来自 replay-backed receipts；verifier result path 使用 single `HandoffPackagePayload` load 和 package payload validation，并在需要 replay 时只 clone `final-branch.bundle` 一次，在同一个 clone 中完成 share-safety scan、receipt replay 和 count/hash comparison。Codex 不应手写 handoff evidence，也不应把 raw grep output、full pytest log、完整 stdout/stderr、wheel、sdist、完整 review transcript 或本地 `.run/local/**` 产物打入 handoff。

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
两者彼此正交。`performance` 是轻量性能和治理工具 runtime evidence profile。
性能优化依据实际耗时画像、slowest tests/files 和 before/after 对比，不写
硬编码秒数阈值，不新增独立性能 gate 文件。

正式 profile runner 位于 `tests/tools/run_profile.py`。执行示例：
`conda run -n <environment.yml name> python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast`。
`profile_smoke` handoff validation row 只来自 runner 的 `profile-smoke` profile；
该 profile 是 handoff mandatory smoke 范围，不代表完整 architecture 或 agents
sweep。

`.run/` 只保存输出、profile、临时文件和 handoff/review artifacts，不作为
tracked tool source。普通 `python -m pytest` 是开发者默认非重型 suite，
默认排除 `integration`、`mutation`、`slow`、`performance` 和
`requires_local_data`；正式 evidence 使用 `tests/tools/run_profile.py`，
完整收口使用 runner 的 `all` profile 显式选择。

sequential `all` profile（串行全量运行配置）是正式 correctness truth gate（正确性事实门禁）。
`all-xdist` 通过透传 bounded `-n 4 --dist loadfile` 使用 `pytest-xdist`，只作为 dev-only speed gate；
它不替代 sequential `all`，也不改变 profile scope、marker 或 validation status 语义。

## 治理规则事实源

`.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 是内部治理规则事实源。`verify_governance.py`、source-tree scan、bundle share-safety scan、共享安全路径规则和相关 tests 只从这里读取或派生 current-contract-only 规则、版本面规则、negative-test allowlist、source-tree scan allowlist、package safety path parts 和 status token。

边界回归由 owner-layer negative tests 和治理 gate 记录为 `FAIL`，不依赖文案关键词。

## 运行结果与 JSON

`<run_root>/run.json` 是运行级机器事实源，`<run_root>/summary.md` 是运行级人读摘要，实例结果位于 `<run_root>/outputs/<output_id>/`。QRP-owned JSON 由 Contracts 层唯一的内部 strict codec 读写；该 owner 不从 package root 导出。读取必须是 strict UTF-8 顶层 object，并在构造 mapping 前拒绝 duplicate keys，同时拒绝 `NaN`、`Infinity`、`-Infinity`、非字符串 object key、非 JSON-native value、`schema_version` 和 `contract_version`。写入必须 deterministic、`allow_nan=False`，不得使用 `default=str` 或 key coercion。artifact owner 在 strict decode 后校验精确 `schema_id`、exact field set 和业务不变量；版本不编码进 `schema_id`。

Dataset `spec.json` 和 `_manifest/manifest.json` 使用 exact current variant grammar，拒绝 unknown fields，并通过 Storage 唯一的 regular-file/size/identity preflight、bounded strict JSON read 与 same-file identity verification 读取。write-side `DataSpec` 在任何 path 或 Parquet IO 前校验 exact runtime enum、identifier、tuple 和 mapping kinds；DataFrame columns、projection 和 physical filters 在 Pandas/Arrow IO 前使用 exact owner grammar。manifest 复制的 spec fields 使用同一个 typed canonical identity 做精确绑定，不采用 Python 宽松数值相等。计数只接受 `type(value) is int` 的非负值并且 summary counts 不得超过 `row_count`；canonical timestamp date bounds 与 canonical `year=YYYY` partitions 精确交叉校验；status 只接受 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`，不得规范化 lowercase 或 `ok`。Physical Arrow schema identity 递归记录 field name、type、nullability、nested child nullability 以及 dictionary index/value/ordered 参数，并拒绝超出 persisted Parquet physical type domain 的 descriptor 或 nullable map key，再以 typed canonical identity hash 绑定 manifest。normal read 和 fingerprint 对 manifest 已声明的 failed status、duplicate PK、PK null 或 nonfinite fail closed；diagnostic inspection 仍可读取同一结构合法 artifact。Canonical identity 使用显式 type tag 和 length framing；非字符串 mapping key 拒绝，字符串不做 Unicode normalization，Timestamp 与字符串保持不同身份。bounded canonical writer 对 finite Decimal 采用 4096 coefficient-digit indivisible ceiling，并拒绝超出当前 runtime `MIN_EMIN..MAX_EMAX` adjusted-exponent domain 的值；`max_bytes=None` 与 ceiling 内的 canonical bytes 不变。

Dataset fingerprint 只用于快速排除不相等候选，不能单独产生 `MATCH`。fingerprint 对普通列直接 hash，对需要规范化的物理 map 与 float primary key 使用 bounded slices；只有顶层 physical float32/float64 primary key 的 `-0.0`/`+0.0` 采用 exact-key 相等语义，float16、普通值及所有 nested float 保留 bit identity。exact compare 通过同一 `DatasetSession` physical traversal，把 persisted Arrow primary key 与 selected-row value 以单个 bounded append sink 写入 Storage 唯一的临时 SQLite index，再以 bounded missing/extra/changed counts 证明相等。受支持的调用方合同要求两个 artifact 在整个比较过程中保持静止（quiescent）；各阶段的 session reopen 和 identity token 复核只提供尽力而为的已观测漂移检测（best-effort observed-drift detection），不构成左右 artifact 的联合原子快照或稳定点。file/partition layout 只进入 `layout_delta`，不改变 logical-content `MATCH`。比较过程复用一个 Runtime policy、一次只打开一个 dataset session，并把 inventory/index reserve、reader working set、bounded row payload、同等大小的一份 SQLite BLOB binding transient、初始 SQLite page 和逐行写入分别纳入 memory/disk gate。

compare 按 `output_id` 配对，并使用 `output_definition_id` 与 metadata 做 typed-exact 语义校验。它要求 declared `summary.md` 存在，把 run output index 与 persisted output manifest 精确绑定，并验证 dataset/layer date、duplicate/status、validation/data/checks 和 overall/matrix 的交叉不变量。local verification 先发布人读 summary，最后才替换 completed `run.json`；output index 只接纳 bundle containment 内实际存在的 declared files。状态 token 与聚合只由 `policy/check_status.py` 定义，handoff contract 不保留 copied-tool fallback 或平行聚合算法。

REVIEW_SAFE evidence tests 分为纯 synthetic fast gate 和 data/parquet contract gate：

```bash
conda run -n <environment.yml name> python tests/tools/run_profile.py --list-profiles
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile <profile> --run-dir .run/<purpose>/<profile>
```

handoff tests 的具体执行入口统一使用 `tests/tools/run_profile.py`。测试位置按 owner layer 放置；详细 placement 规则由 maintenance references 和 architecture gates 拥有。

handoff builder/verifier 位于 `.agents/skills/quant-research-platform-maintenance/scripts/handoff/`。正式生成入口是 `build_handoff_package.py --repo-root . --out-dir .run/handoffs --task-name <task-name> --base-ref <base-ref>`；builder 只读取当前 clean Git state 和这些 CLI options，写入 bundle，从 bundle clone 自动生成 machine receipts 和 validation summary，再通过 package payload validation 检查 schema、hash 和 share-safety。`verify_handoff_package_result()` 是唯一 package-level verifier function；standalone verifier CLI 保留为完整 bundle replay 和 receipt comparison 的直接入口，未经用户明确批准不新增 wrapper、alias、adapter 或 compatibility layer。builder 不运行独立 safety scan；share-safety status 和 zip sha256 来自同一个 package payload validation result。Standalone verifier 从 `final-branch.bundle` clone replay，一次性复用 `SourceTreeScanInput` 驱动的 source-tree scan 结果执行 share-safety 和 hard-scan receipt replay，并比较 packaged receipts、validation summary 和 manifest。具体 layout、schema、receipt 和 active denied semantics 由 executable contracts、verifier 和 tests 拥有。
