# QRP

QRP 是本地量化研究平台，用于数据合同、因子与特征研究、评估、模型训练、预测、组合与历史 execution simulation、报告和 evidence。它不是 live trading、paper account trading 或 broker/exchange execution system，也不提供 investment advice。

## 目录与 owner

```text
src/         QRP production code
tests/       当前行为、合同与架构测试
docs/        当前合同文档
data/        本地数据占位；不提交 private data
.run/        ignored local inputs/outputs/evidence；不定义代码事实
```

- `sources/` 负责摄取、raw schema 和 raw/interim 转换；`markets/` 负责市场领域实现。
- `contracts/` 定义结构合同；`policy/` 定义共享安全规则；`profiles/` 绑定 API、CLI 与 workflow 组合根。
- `research/` 和 `compute/` 负责研究与数值计算；`storage/` 负责 dataset IO、persisted identity、exact compare 与 data layout。
- `runtime/` 负责资源准入与 execution metrics；`artifacts/` 写 LOCAL_ONLY/REVIEW_SAFE 文件；`reports/` 构建内容。
- `workflows/` 只编排；`interfaces/` 暴露已批准的 API/CLI。

## Managed operation

一个 managed data-scale operation 先取得 host-local lock，只捕获一次 `available0`，默认 planning target 约为 `available0` 的 70%；explicit cap 只能进一步降低它。每个数据规模 read/decode/allocation 在发生前做保守准入，流程保持 bounded；aggregate persisted output 可以超过 RAM budget 而 exact 完成。主要 workflow stage 按顺序执行，前一 stage 释放 transient state 后，后一 stage 复用同一个 operation budget；只为当前 stage 中真正同时存活的 allocation 准入。新 payload/workspace/output 只有一个 logical owner，在 producer/consumer 边界转移，不重复计费；owner-local context 退出时释放资源。

默认 single-process + native threads，只使用一层主要并行，线程数不超过物理核并可因 planning target 下调。70% 及派生出的 `memory_budget_bytes` 都只是相对安全的 soft planning reference，不是进程 RSS 或真实 allocation 的硬上限；lease bytes 是可相加的保守粗估，不要求字节级精确。约 80% 只描述工程上的大致安全包络，也不是第二硬阈值。仓库外 fresh-child 的 RSS/CPU/I/O 观测只用于校准和验收，不是 runtime planning authority。小幅、稳定且可解释的估算或包络 overshoot 应标为风险供用户判断，不在 70% 或 80% 的精确点自动判死；borrowed baseline、native workspace 和这些观测都不参与 production planning 或重复计费。

## Storage and publication

Dataset identity 由 persisted schema、dtype、PK、logical window 与 exact content 定义。Fingerprint 只用于 quick reject；只有 exact compare 可以产生 `MATCH`。输出在 transaction-local disk preflight 后写入同父目录 UUID staging，再通过平台 native no-replace 或 owner-qualified replace 发布；成功才 atomic publish，失败保留 old-good，并只清理本次 operation 能证明创建的资源。存在 primary `BaseException` 时，释放或验证失败不得覆盖它。

QRP 采用 current-contract-only：替换旧实现时删除旧入口，不保留 alias、compat reader 或平行实现。

## Public surface

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements

qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

新增 public API/CLI、依赖、framework、connector 或 deployment 需要明确批准。

## 直接验证

环境名以 `environment.yml` 的 `name` 为准：

```text
conda run -n <env> ruff check .
conda run -n <env> pytest -q
conda run -n <env> python -m build --no-isolation --sdist --wheel --outdir <run-dir>
```

默认 `pytest -q` 运行全部测试。Package acceptance 必须从仓库外 clean export 构建 fresh sdist，再从解出的 sdist 构建 wheel，在仓库外 isolated environment 安装，最后从 empty CWD 做 CLI/API smoke；editable checkout 或直接从源码路径导入不算通过。

## 数据安全

不提交 private data、secret、vendor payload、本地绝对路径、股票代码样本、row-level sample 或 raw traceback。REVIEW_SAFE evidence 只包含 schema、计数、hash、状态、逻辑路径结构和不含私有行的摘要。
