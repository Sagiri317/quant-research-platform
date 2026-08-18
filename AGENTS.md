# QRP Agent Instructions

本文件只保存长期、可执行的仓库边界。当前 branch、HEAD、Phase、finding、测试结果和 next action 只写入 `.run/ledger.md`。

## 1. Authority and live truth

规范优先级：

```text
用户本轮明确指令
-> active Goal
-> live AGENTS.md
-> GPT 项目源长期工程标准
```

事实优先级：

```text
live Git / worktree / index
-> live code / tests / config / schemas / runtime behavior
-> 当前 Ledger 与当前 commit 的实际证据
-> 历史 branch、bundle、handoff、review、报告仅作线索
```

Live implementation 描述“现在是什么”，不自动决定“应该是什么”。发现 Goal 与实现冲突时，先在 Ledger 登记 drift，再按 Goal 处理。不得覆盖无法解释的 WIP。

## 2. Product boundary and preserved capability

QRP 是本地量化研究平台：

```text
source / market data
-> interim / processed dataset
-> factor / label / mask / context
-> research dataset / transform
-> evaluation / model / prediction
-> portfolio / diagnostics
-> artifact / report
```

默认不属于：

```text
live trading / paper trading
broker / exchange connector
SaaS / production deployment
investment advice
```

除非用户明确批准，不新增 public API/CLI、scheduler、worker、deployment、broker/vendor connector、新市场垂直链路、新长期架构 owner 或新重型 runtime dependency。

不得为了删代码、降内存、缩依赖、架构对称或测试方便而删除 active Goal base 上的量化能力、已注册模型、scenario、schema、artifact/report、A-share/CSMAR、既有 Binance/Crypto 能力或 Goal 指定的 mandatory dependencies。

未经用户明确批准，不得静默改变：

```text
factor / label / mask definitions
point-in-time availability
schema / dtype / primary key
logical time / window / horizon / output scope
train-validation-test separation
execution timing / lag / cost / return attribution
prediction determinism
persisted exact identity
approved public machine contracts
```

确认 semantic bug 时可以修复，但必须先有最小 executable RED、手算 fixture 或足以唯一证明违反合同的静态证据，并验证 persisted/downstream behavior。

## 3. Ponytail Ultra

使用本地已经安装的官方 Ponytail plugin/skill。不得重新安装、升级、修改 hooks/plugin、复制 skill 正文进仓库，或声称执行了未实际执行的 audit/review。

理解真实调用流后，按第一个可行层停止：

```text
需求能否不存在？
-> 能否删除旧路径或旧 surface？
-> 能否复用现有唯一 owner？
-> stdlib / Git / pytest / 平台原生能力？
-> 已安装依赖？
-> 能否内联？
-> 最后才新增最小实现
```

优先挑战并删除：

```text
dead / tests-only production surface
无独立 invariant 的 wrapper / alias / shim / fallback
旧实现与新实现并存
重复 validator / planner / estimator / registry / schema constants
proof / freeze / receipt / handoff / hash ceremony
当前调用流 code-map / code-flow 快照
private helper、文件名、import、__all__ shape-only tests
无 live caller 的 platform / concurrency / hostile-filesystem machinery
DataFrame -> Python row graph -> DataFrame
repeated concat / vstack / copy / full materialization
```

以下不得以“代码更少”为由删除：

```text
获批量化功能与模型
private-data boundary
one-shot memory admission 与共享 root budget
persisted schema / PK / exact compare
atomic old-good preservation
truthful partial state
primary exception preservation
owner-qualified cleanup
真实 public / trust / platform boundary
```

Ponytail 的目标是删除 accidental complexity，不是删除所有 abstraction。移除 abstraction 前必须证明不会丢失 public/trust/lifecycle/ownership/platform boundary。

不维护全仓 KEEP manifest，不逐文件证明“永远不可删”。Repo-wide audit 只在 active Goal 指定的 intake/final 节点运行；普通 Phase 只审 committed diff 和 affected owners。

## 4. Owner and scope discipline

QRP 使用 current-contract-only：live behavior、approved machine contract、owner tests 与 dependency direction 定义当前合同；历史名称、目录或实现形状不自动构成兼容义务。

同一 invariant 只允许一个权威 owner；“唯一 owner”表示唯一规则来源，不表示把所有逻辑塞进一个大文件。

| Semantics | Owner |
|---|---|
| base contracts、IDs、time windows、validation | `contracts/` |
| provider/raw ingestion、decode、normalization | `sources/` |
| calendar、market rules、valid bars、masks | corresponding market implementation |
| shared safety rules、PASS/FAIL/NOT_RUN/BLOCKED aggregation | `policy/` |
| approved profile/composition-root binding | `profiles/` |
| queryable catalog views over owner-defined facts | `catalog/`；不得重定义业务语义或复制 registry |
| factor/label/dataset/model/prediction/portfolio numerical semantics | `research/` / `compute/` |
| physical IO、manifest、persisted identity、exact compare | `storage/` |
| one-shot admission、root budget、reservation、RSS evidence | `runtime/` |
| report content | `reports/` |
| artifact validation/write | `artifacts/` |
| run identity、orchestration、publication、partial truth | `workflows/` |
| approved adapters | `interfaces/` |

每个 writer Phase 开始时，只允许一次 bounded discovery pass，并在 Ledger 冻结当前 Phase 的精确 path/symbol allowlist、acceptance ID、直接证据、test path 和 forbidden adjacent owners。禁止 `src/**`、`tests/**` 等仓库级修改范围。

Allowlist 外编辑前必须停止并在 Ledger 写明：具体 caller、失败证据、拟加入路径和原因。只有同一 acceptance 的直接 caller/sibling/downstream 可以在 Ledger 中扩展后继续；新 owner、新产品、新 public surface、新 dependency 或新平台进入 `DECISION_REQUIRED`。

每项 production/config 修改前必须绑定当前 HEAD 的 executable RED 或 conclusive static proof。只因“更干净、更通用、未来可能需要”不得修改。

不得新增 tracked `governance/`、`status/`、`audit/`、`review/`、`handoff/`、`freeze/`、`receipt/`、`code-flow/`、control generator 或 evidence archive。动态 Goal/Ledger/Reviewer 只存在于 ignored `.run/`。

## 5. One-shot local resource contract

Composition root 仅指能够独立启动 data-scale work 的已批准 CLI/public
entrypoint 或现有 profile/workflow root。普通 helper、tests-only harness、
只能在父 root 内调用的 stage 不是 root。

每个 data-scale root 在任何 data-scale read/decode/allocation 前：

```text
1. nonblocking acquire host-local cross-process data-scale lock
2. capture available0 exactly once
3. memory_budget_bytes = min((available0 * 70) // 100, explicit_cap_bytes)
4. create one explicit RuntimeResourcePolicy + RAM lease ledger
```

`explicit_cap_bytes` 必须是正整数 bytes，`bool` 不合法，且只能降低预算。
Provider failure、`available0 <= 0`、非法 cap 或最终 budget `<= 0` 必须在
data-scale action 前稳定失败。生产 memory facts 只允许：
`captured_available_memory_bytes`、`memory_budget_bytes`、
`used_memory_bytes`、`available_memory_bytes`。

一次 operation 只有一个 immutable root identity/ledger，贯穿
source、dataset、research、model、prediction、portfolio、publication 和
cleanup。每个 logical live RAM payload/workspace 只有一个 transferable
`MemoryLease`；producer 在 allocation 前 acquire/grow，在 conservative actual
bound 后 shrink，跨边界只 transfer，不重复收费，最终 close。Root 创建前已
存在的 public DataFrame/ndarray 是 borrowed baseline，只计 operation 内新
copy/workspace/output。

默认本地 single-process + native threads，只允许一层主要并行。私有
data-scale helper 必须显式接收 policy；不得通过 ambient discovery 创建或继承
root。Root state 只有 `ACTIVE`、`CLOSED`、`POISONED`。每个 exit 都 LIFO close
registry，验证 RAM/disk/thread/child/publication state 为零，并保留 primary
`BaseException`；不可清理资源使 root/process `POISONED` 并阻止第二个
data-scale root。

生产 safety authority 是 admission、gross simultaneous-live bound、lease
lifetime 和 machine lock。RSS 只属于 repository-external fresh-child release
evidence：以 `available0` 为分母，`<=80%` 为 PASS candidate，`80%-82%` 为
AT_RISK，`>82%` default FAIL；这些阈值不得进入 production planning。

## 6. Quant semantic discipline

- Label timing、signal observation、entry、exit、portfolio attribution 必须用少量有效 bar 的手算 fixture 明确。
- 防泄漏从 live label lookahead 推导，复用现有 purge/split contract；feature fit、model fit、model selection 不得读取 future/validation/test rows。
- Prediction artifact 与 portfolio/research 输入按 persisted primary key 对齐；required coverage、允许缺失和 failure mode 必须明确，不得从 context column 猜 prediction。
- Missing/duplicate/misaligned required prediction 必须稳定失败。
- Fixed seed、backend version、thread settings 与现有 exact/tolerance contract 决定 determinism；不得承诺无法证明的跨平台 bitwise identity。
- Diagnostics/proxy 必须诚实命名。未实现的 impact/risk/capacity model 不得输出零值并标称已生效。
- Mandatory model dependencies 和 base-registered algorithms 不得通过 silent fallback、dummy backend 或 optionalization 逃避验证。

## 7. Performance discipline

先保证量化语义和硬内存，再优化：

```text
column projection / predicate / partition pushdown
columnar batches instead of Python row graphs
bounded Arrow/Parquet batch + readahead
incremental writer and accumulator
avoid repeated concat / vstack / full copies
reuse one preparation across compatible model candidates
one parallelism layer; bound BLAS/OpenMP/joblib/XGBoost/LightGBM threads
```

Native thread environment 必须在 measured fresh child 导入相关 native libraries 前设置。Pandas Copy-on-Write 下，只在 mutation tests 证明安全后删除 defensive copy。

不得仅为“可能更快”增加 Dask/Ray/Modin/Polars/DuckDB。Profiling 工具可临时装入隔离 dev 环境，但不得进入 runtime dependency metadata。

## 8. Identity, publication, cleanup and private safety

- Persisted schema、dtype、PK、lineage 和 exact equality 以最终 persisted representation 为准。
- QRP-owned generated JSON 的顶层 identity 继续使用 live current contract 的 `schema_id`；不得为重构方便引入平行 version/type identity。
- Fingerprint/hash 只能 quick reject；只有 exact comparison 可以产生 `MATCH`。
- 新输出在同父目录 owner-owned staging 中完成和验证，再 atomic publish；失败保留 old-good。
- Partial run 必须真实标记。
- Cleanup 只能删除当前 owner 可证明创建的资源。
- Active `BaseException` 存在时，cleanup/verify/release failure 不得覆盖 primary；无 primary 时 secondary 正常传播。
- Agent/reviewer 不得把 private row values、raw payload、secret、未经清洗的本地绝对路径或 raw traceback 写入 Ledger、review、retained evidence 或 final report。证据只保留 schema、计数、hash、状态、逻辑路径别名和无私有行摘要。
- 默认 threat model 是可信本地用户与正常并发；hostile local actor、reparse/hardlink/native namespace 防御需要用户明确批准。

## 9. Implementation, review and correction

任一时刻只有一个 writable agent。主 Goal agent可以计划、实现、验证、commit/push 和 reconciliation，但不得担任自己的 formal reviewer。

Reviewer 必须 fresh、read-only，只读 committed snapshot、Goal、Ledger、diff 与必要 affected owners/callers/downstream；不读取主 agent 私有分析、其他 reviewer 报告或辩护说明。Formal review 只写 reviewer-specific、仓库外 temp/cache。Review window 内主 agent暂停 tracked files、commit/branch、push 与 Ledger 写入。

每个 Phase 的 review/correction 次数由 Goal 限定。Blocking finding 必须：修复并由 fresh reviewer recheck；或用直接证据反证并由 fresh reviewer recheck；或进入 `DECISION_REQUIRED`。不得静默省略、降级、改写或以多数票关闭。

Review 使用 clean committed SHA。Commit SHA 即 snapshot，不创建 freeze packet、receipt、inventory hash、ZIP 或 handoff verifier。

## 10. Tests, package, Git and resume

测试原则：最少，但能杀死错误实现。优先保留 timing/leakage fixture、chunk/full/spill parity、schema/PK/exact identity、one-shot/shared budget、all-model tiny smoke、prediction-backed portfolio、atomic old-good、primary exception 和 private-safe failure。优先删除 private helper、文件/import/`__all__` shape、coverage-only 和 retired-mechanism tests。

Formal tests/build/benchmark 只写仓库外 owner-qualified temp/scratch/benchmark roots，不覆盖 canonical user data、old-good artifacts 或 tracked files。使用 Conda 时，环境名从 live `environment.yml` 动态读取，不硬编码。Direct full suite 必须显式覆盖项目默认的 non-slow selection；普通 `pytest -q` 不得冒充 full。

Package validation 从 exact committed HEAD 的仓库外 clean export 构建 fresh sdist，再从 extracted sdist 构建 wheel，并在 repo 外隔离导入；不得由 editable checkout 假通过。

Git 安全：

```text
不 force push / rebase / amend
不 merge/push main
不 reset/restore/clean/stash 用户或其他 agent 工作
不做递归/通配删除
每个 coherent commit 按授权 push feature branch
.run ignored，永不 stage/commit
```

新 thread、上下文压缩、模型切换或恢复后，从 byte 1 重读 `AGENTS.md`、`.run/goal.md`、`.run/ledger.md`，核对 branch/HEAD/status、Ponytail ultra、当前 Phase scope、review quota 和唯一 next action。不得无理由重跑已关闭的 repo-wide audit 或 full suite。
