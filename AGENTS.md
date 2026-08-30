# QRP Agent 说明

本文件只保存长期、可执行的仓库边界。一次性状态、运行证据和当前改动以 live Git、代码、测试、配置以及 ignored `.run/` 记录为准。修改前读取目标文件、适用规则和 `git status`，保留用户及其他 agent 的已有改动。

## 权威与产品边界

任务边界按明确任务范围 -> ignored `.run/` 相关控制文件 -> `AGENTS.md` -> 工程规范绑定；当前代码/测试/配置/运行时只提供当前事实。历史提交和旧报告只能作为线索，不替代当前证据；不得覆盖无法解释的 WIP。

QRP 是本地量化研究平台：

```text
source / market data
-> interim / processed dataset
-> factor / label / mask / context
-> research dataset
-> evaluation / model / prediction
-> portfolio / diagnostics
-> artifact / report
```

它不属于实盘交易、模拟交易账户、broker/exchange、SaaS、部署或投资建议。除非明确批准，不新增 public API/CLI、scheduler/worker、connector、产品垂直、长期架构 owner 或重型 runtime dependency。不得为了删代码、降内存、缩依赖或测试方便删除已批准的量化能力、模型、scenario、schema、artifact/report、A-share/CSMAR、Binance/Crypto 或 mandatory dependencies。

未经明确批准，不得静默改变 factor/label/mask 定义、point-in-time availability、schema/dtype/PK、logical window/horizon/output scope、train-validation-test 分离、timing/lag/cost/return attribution、prediction determinism、持久化 identity 或已批准的 machine contract。确认语义 bug 需要可执行 RED、手算 fixture 或唯一的静态证据，并验证持久化及下游行为。

## Ponytail 与最小设计

使用本地已经安装的官方 Ponytail Ultra；不得安装、升级、复制、仿写或声称执行未实际运行的工具。理解真实调用流后，按第一个可行层停止：

```text
需求能否不存在？
-> 删除旧路径或旧 surface？
-> 复用已有 owner？
-> stdlib / Git / pytest / 平台原生能力？
-> 已安装 dependency？
-> 内联？
-> 最后才新增最小机制
```

优先删除无用或测试专用的 surface、无独立 invariant 的 wrapper/alias/shim/fallback、重复 owner/materialization、无当前 caller 的平台/并发机制和仅检查 shape 的测试。不要增加投机性 abstraction、第二套 source of truth 或不必要的 dependency。删减不得削弱 private-data boundary、量化语义、资源准入、有界完成、exact identity、old-good、primary exception、owner-qualified cleanup 或真实 public/trust boundary。

## 语义 owner

同一 invariant 只允许一个权威 owner；唯一 owner 表示唯一规则来源，不要求把所有逻辑塞进一个文件。当前 owner 关系如下：

| 语义 | Owner |
|---|---|
| 基础合同、IDs、时间窗口、校验 | `contracts/` |
| provider/raw 摄取、解码、标准化 | `sources/` |
| 日历、市场规则、有效 bars、masks | 对应市场实现 |
| 共享安全规则与结果汇总 | `policy/` |
| profile/composition-root 绑定 | `profiles/` |
| 基于 owner 定义事实的 catalog 视图 | `catalog/` |
| factor/label/dataset/model/prediction/portfolio 语义 | `research/` / `compute/` |
| 物理 IO、manifest、持久化 identity、exact compare | `storage/` |
| 准入、root budget、reservation、RSS 证据 | `runtime/` |
| artifact 校验/写入 | `artifacts/` |
| 报告内容 | `reports/` |
| 编排、发布、部分真实状态 | `workflows/` |
| 已批准路由与适配器 | `interfaces/` |

仓库采用 current-contract-only：旧名称、目录和实现形状不自动构成兼容义务；迁移完成后删除旧入口，不保留平行实现。

## 托管操作与性能

每个托管的数据规模操作只有一个简单资源边界：先取得 host-local data-scale lock，只捕获一次 `available0`，默认规划目标为 `min(available0 * 70%, explicit cap)`。70% 和派生出的 `memory_budget_bytes` 都是相对安全的软规划参考，不是进程 RSS、真实 allocation 或逻辑内存的硬上限。lease 以 bytes 表达只是为了让 owner 的保守粗估可相加比较，不要求或声称字节级精确。进入数据规模 read/decode/allocation 前按大致工作集估算选择有界批次；只有保守估算表明无法维持相对安全或 explicit cap 明确限制时才稳定失败，不因轻微估算或观测越过 70% 的精确点自动判死，也不建立第二套硬阈值。聚合持久化输出可以超过 RAM 规划目标而 exact 完成。已有 operation 内的 public DataFrame/ndarray 是 borrowed baseline；新 payload/workspace/output 只有一个 logical owner，跨 producer/consumer 转移而不重复计费，离开 owner context 时关闭。borrowed baseline、native workspace 与仓库外 RSS 观测不得重复计费。

默认使用 single-process + native threads，只允许一层主要并行；线程数不超过物理核并可因预算下调。不得叠加 Python worker pool、native fan-out、scheduler 或 executor。native thread environment 必须在 measured fresh child 导入相关库前设置。

发布使用 transaction-local 保守磁盘预检、同父目录 UUID staging 和平台 native no-replace 或 owner-qualified replace。成功才 atomic publish，失败保留 old-good；只清理本次 operation 能证明创建的资源。存在 primary `BaseException` 时，cleanup/verify/release failure 不得覆盖它。

Persisted identity 以 schema、dtype、PK、lineage 和 exact equality 为准；fingerprint 只作 quick reject，partial output 必须真实标记。

性能优先使用列投影、predicate/partition pushdown、列式有界批次和增量输出，避免重复 concat、vstack、full copy 与 Python row graph；不为可能的收益增加重型 dependency。

## 量化与私有数据安全

- Label timing、signal observation、entry、exit、portfolio attribution 用少量有效 bar 的手算 fixture 明确；防泄漏复用 live lookahead 与 purge/split contract。
- Prediction artifact 与 portfolio/research 输入按 persisted PK 对齐；missing、duplicate、misaligned required prediction 必须稳定失败。
- Fixed seed、backend version、thread settings 与既有 exact/tolerance contract 决定 determinism；不得承诺无法证明的跨平台 bitwise identity。
- Diagnostics/proxy 必须诚实命名；未实现的模型不得输出零值并声称已生效。Mandatory model dependency 和 base-registered algorithm 不得通过 silent fallback、dummy backend 或 optionalization 逃避验证。
- 不打印、上传或保留私有行值、raw payload、secret、未经清洗的本地绝对路径或 raw traceback。共享证据只包含 schema、计数、hash、状态、逻辑路径别名和不含私有行的摘要。

## 检查与 Git

- 修改后运行覆盖本次行为的最窄、非破坏性检查，并通过 `python -m ruff format --check .`；默认 `pytest -q` 运行全部测试。
- Package acceptance 从仓库外 clean export 开始：构建 fresh sdist，从解出的 sdist 构建 wheel，在仓库外隔离环境安装，再从 empty CWD 做 CLI/API smoke；editable checkout 或直接从源码路径导入不算通过。
- 不 force push、rebase、amend、reset、restore、clean、stash；仅在明确任务范围授权时合并到 main；不覆盖其他 agent 的改动。`.run/` 为被忽略的本地状态，永不 stage/commit。
