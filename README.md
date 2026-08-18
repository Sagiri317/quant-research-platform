# QRP

QRP 是本地量化研究平台，用于数据合同、因子与特征研究、模型训练、预测、评估、组合构建、报告和 evidence。它不是生产或模拟交易系统，不接 broker/exchange，不提供 investment advice。

## 现场事实

仓库文档只描述长期合同。当前 branch、phase、review 与运行结果从 live Git 和本轮任务读取：

```text
git status --short --branch
git log -1 --date=iso-strict --format="%H%n%cd%n%s"
```

Agent 执行规则见 `AGENTS.md`。

## 目录与 owner

```text
src/         QRP production code
tests/       当前行为、合同与架构测试
docs/        当前合同文档
data/        本地数据占位；不提交 private data
.run/        ignored local outputs/evidence；不定义代码事实
```

- `sources/` 负责摄取、raw schema 和 raw/interim 转换。
- `markets/` 负责市场领域与具体市场实现。
- `contracts/` 定义结构合同；`policy/` 定义共享安全规则。
- `profiles/` 是 API、CLI 与 workflow 的组合根。
- `research/` 和 `compute/` 只做研究与数值计算。
- `storage/` 拥有 dataset IO、persisted identity、exact compare 与 data layout。
- `runtime/` 拥有 resource policy 与 execution metrics。
- `reports/` 构建内容；`artifacts/` 写 LOCAL_ONLY/REVIEW_SAFE 文件。
- `workflows/` 只编排；`interfaces/` 暴露已批准 API/CLI。

## 核心合同

一个 data-scale operation 先取得 host-local cross-process lock，再只捕获一次 available memory，生产预算为其 70% 与 explicit cap 的较小值。整个 operation 共享一个 explicit root policy、RAM lease ledger、transferable ownership 和 cleanup registry；每次数据规模 allocation 前先 acquire/grow lease，跨 producer/consumer 边界只 transfer，最终 close。RSS 只由外部 fresh-child release evidence 观测，80%-82% 是 release 阈值，不是生产预算。

Dataset identity 由 persisted schema、dtype、PK、logical window 与 exact content 定义。Fingerprint 只用于 quick reject；只有 exact compare 可以产生 `MATCH`。Publication 必须 atomic、保留 old-good，cleanup 只处理 owner-qualified resources，active `BaseException` primary 不被 secondary failure 覆盖。

QRP 采用 current-contract-only：替换旧实现时删除旧入口，不保留 alias、compat reader 或平行实现。

## 运行结果

```text
<run_root>/
  run.json
  data/{interim,processed,research}/
  outputs/<output_id>/manifest.json
  logs/
  summary.md
```

`run.json` 是运行级 machine truth，`summary.md` 是人读摘要。Dataset 使用 `spec.json` 与 `_manifest/manifest.json`。QRP-owned JSON 顶层是 object，并使用不带版本后缀的 `schema_id`。

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
conda run -n <env> python -m pytest -q
conda run -n <env> python -m pytest -q -o "addopts=--import-mode=importlib --strict-markers" tests
conda run -n <env> python -m build --no-isolation --sdist --wheel --outdir <run-dir>
```

默认 pytest 只排除明确标记的 `slow` tests；完整命令覆盖默认 `addopts`。Build smoke 解压 wheel 后用 `PYTHONPATH` import，不安装 package。

## 数据安全

不提交 private data、secret、vendor raw marker、本地绝对路径、股票代码样本、row-level sample 或 raw traceback。REVIEW_SAFE evidence 只包含 schema、计数、hash、状态、逻辑路径结构和不含私有行的摘要。
