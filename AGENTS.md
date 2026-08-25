# QRP Agent Instructions

本文件只保存长期、可执行的仓库边界。一次性状态、运行证据和当前改动以 live Git、代码、测试、配置以及 ignored `.run/` 记录为准。修改前读取目标文件、适用规则和 `git status`，保留用户及其他 agent 的已有改动。

## Authority and product boundary

任务边界按 explicit task envelope -> ignored `.run/` 相关控制文件 -> `AGENTS.md` -> engineering standard 绑定；live code/tests/config/runtime 只提供当前事实。历史提交和旧报告只能作为线索，不替代当前证据；不得覆盖无法解释的 WIP。

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

它不属于 live trading、paper trading、broker/exchange、SaaS、deployment 或 investment advice。除非明确批准，不新增 public API/CLI、scheduler/worker、connector、产品垂直、长期架构 owner 或重型 runtime dependency。不得为了删代码、降内存、缩依赖或测试方便删除已批准的量化能力、模型、scenario、schema、artifact/report、A-share/CSMAR、Binance/Crypto 或 mandatory dependencies。

未经明确批准，不得静默改变 factor/label/mask definitions、point-in-time availability、schema/dtype/PK、logical window/horizon/output scope、train-validation-test separation、timing/lag/cost/return attribution、prediction determinism、persisted identity 或 approved machine contract。确认 semantic bug 需要 executable RED、手算 fixture 或唯一的静态证据，并验证 persisted/downstream behavior。

## Ponytail and minimal design

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

优先删除 dead/test-only surface、无独立 invariant 的 wrapper/alias/shim/fallback、重复 owner/materialization、无 live caller 的 platform/concurrency machinery 和 shape-only tests。不要增加 speculative abstraction、第二套 source of truth 或不必要的 dependency。删减不得削弱 private-data boundary、量化语义、资源准入、bounded completion、exact identity、old-good、primary exception、owner-qualified cleanup 或真实 public/trust boundary。

## Semantic owners

同一 invariant 只允许一个权威 owner；唯一 owner 表示唯一规则来源，不要求把所有逻辑塞进一个文件。当前 owner 关系如下：

| Semantics | Owner |
|---|---|
| base contracts、IDs、time windows、validation | `contracts/` |
| provider/raw ingestion、decode、normalization | `sources/` |
| calendar、market rules、valid bars、masks | corresponding market implementation |
| shared safety rules and outcome aggregation | `policy/` |
| profile/composition-root binding | `profiles/` |
| catalog views over owner-defined facts | `catalog/` |
| factor/label/dataset/model/prediction/portfolio semantics | `research/` / `compute/` |
| physical IO、manifest、persisted identity、exact compare | `storage/` |
| admission、root budget、reservation、RSS evidence | `runtime/` |
| artifact validation/write | `artifacts/` |
| report content | `reports/` |
| orchestration、publication、partial truth | `workflows/` |
| approved routes and adapters | `interfaces/` |

仓库采用 current-contract-only：旧名称、目录和实现形状不自动构成兼容义务；迁移完成后删除旧入口，不保留平行实现。

## Managed operation and performance

每个 managed data-scale operation 只有一个简单资源边界：先取得 host-local data-scale lock，只捕获一次 `available0`，默认预算为 `min(available0 * 70%, explicit cap)`。进入数据规模 read/decode/allocation 前做保守准入，无法满足时稳定失败；生产流程必须 bounded，并且 aggregate persisted output 可以超过 RAM 预算而 exact 完成。已有 operation 内的 public DataFrame/ndarray 是 borrowed baseline；新 payload/workspace/output 只有一个 logical owner，跨 producer/consumer 转移而不重复计费，离开 owner context 时关闭。

默认 single-process + native threads，只允许一层主要并行；线程数不超过物理核并可因预算下调。不得叠加 Python worker pool、native fan-out、scheduler 或 executor。native thread environment 必须在 measured fresh child 导入相关库前设置。

发布使用 transaction-local conservative disk preflight、同父目录 UUID staging 和平台 native no-replace 或 owner-qualified replace。成功才 atomic publish，失败保留 old-good；只清理本次 operation 能证明创建的资源。存在 primary `BaseException` 时，cleanup/verify/release failure 不得覆盖它。

Persisted identity 以 schema、dtype、PK、lineage 和 exact equality 为准；fingerprint 只作 quick reject，partial output 必须真实标记。

性能优先使用 column projection、predicate/partition pushdown、columnar bounded batches 和 incremental output，避免重复 concat、vstack、full copy 与 Python row graph；不为可能的收益增加重型 dependency。

## Quant and private-data safety

- Label timing、signal observation、entry、exit、portfolio attribution 用少量有效 bar 的手算 fixture 明确；防泄漏复用 live lookahead 与 purge/split contract。
- Prediction artifact 与 portfolio/research 输入按 persisted PK 对齐；missing、duplicate、misaligned required prediction 必须稳定失败。
- Fixed seed、backend version、thread settings 与既有 exact/tolerance contract 决定 determinism；不得承诺无法证明的跨平台 bitwise identity。
- Diagnostics/proxy 必须诚实命名；未实现的模型不得输出零值并声称已生效。Mandatory model dependency 和 base-registered algorithm 不得通过 silent fallback、dummy backend 或 optionalization 逃避验证。
- 不打印、上传或保留 private row values、raw payload、secret、未经清洗的本地绝对路径或 raw traceback。共享证据只包含 schema、计数、hash、状态、逻辑路径别名和不含私有行的摘要。

## Checks and Git

- 修改后运行覆盖本次行为的最窄、非破坏性检查；默认 `pytest -q` 使用项目的 non-slow 选择，完整测试必须显式覆盖默认过滤器。
- Package acceptance 从仓库外 clean export 开始：构建 fresh sdist，从解出的 sdist 构建 wheel，在仓库外隔离环境安装，再从 empty CWD 做 CLI/API smoke；editable checkout 或直接从源码路径导入不算通过。
- 不 force push、rebase、amend、reset、restore、clean、stash 或合并到 main；不覆盖其他 agent 的改动。`.run/` 为 ignored local state，永不 stage/commit。
