# QRP

QRP 是一个本地量化研究平台，用于组织数据合同、因子研究、特征处理、模型训练、预测、评估、组合构建、报告与 evidence。它面向研究和演示，不是生产交易系统，不接 broker 或 exchange，不提供 investment advice。

## 当前定位

QRP 的长期定位是：用严格合同、可复现 evidence 和受控 agent workflow 支撑量化研究迭代。仓库文档只描述长期合同和操作规则，不记录当前分支、当前阶段、当前 PR、最近运行结果或下一步任务。

当前状态必须从 git 和当前任务现场读取：

```bash
git status --short --branch
git log -1 --date=iso-strict --format="%H%n%cd%n%s"
```

## 目录概览

```text
.agents/     agent 指令、skills、governance scripts、handoff tooling
src/         QRP 源码
tests/       当前合同测试、治理测试、架构测试
docs/        当前合同文档和代码地图
data/        本地数据占位说明，不提交私有数据
.run/        本地运行输出和 evidence，不作为 tracked tooling source
```

核心 owner 边界：

- `sources/`：数据源摄取、raw schema、字段映射和 raw/interim 转换。
- `markets/`：通用市场领域与 session window；A 股实现位于 `markets/implementations/a_share/`。
- `policy/`：共享安全规则、ID 安全判断和 PASS/FAIL/NOT_RUN/BLOCKED 状态聚合。
- `contracts/`：结构合同、枚举、ID、时间窗口和基础校验，不保存共享安全规则事实。
- `catalog/`：generic registry class。
- `profiles/`：API、CLI、agent 和 workflow 共用组合根，组装 catalog/source/market/materializer。
- `research/`：通用研究计算，只构建 result，不写文件，不依赖具体市场实现或 sources。
- `compute/`：dataframe/tensor、portfolio math 和 performance 数值核心。
- `storage/`：只负责 dataset IO 与 data/raw、data/interim、data/processed、data/research layout。
- `runtime/`：resources、logging、workflow records 和 execution metrics payload，不写 report artifact。
- `reports/`：只 build/render evidence/report 内容，不写文件。
- `artifacts/`：只负责把 LOCAL_ONLY run outputs 和 REVIEW_SAFE evidence 写成文件。
- `workflows/`：只编排 research -> reports -> artifacts。
- `.agents/`：治理、handoff 和 prompt tooling，不承载 runtime 业务逻辑。

## 当前 public surface

当前 public API 只有：

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

当前 public CLI 只有：

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

## agent 入口

coding agent 应先读 `AGENTS.md`，再按任务范围读取 `.agents/skills/quant-research-platform-maintenance/SKILL.md`。需要深入执行时，再按需读取 `.agents/skills/quant-research-platform-maintenance/references/**`。

`AGENTS.md` 不是状态板。需要当前事实时，必须使用 live git baseline。

## 文档原则

- 文档只保存当前合同。
- open issues 只保存当前 open 或 deferred 问题。
- `.agents` 是 tracked tooling source。
- `.run` 是 output layer 和 evidence layer。
- `.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py` 是内部治理规则事实源，active-doc gate、handoff source-tree scan、single handoff verifier safety scan 和 tests 的 retired/stale/naming/allowlist 规则只能从这里读取或派生；路径名、Python symbol、JSON/YAML/TOML key 和静态字符串合成都由同一规则派生。
- retired governance paths 只要复活即 `FAIL`，不靠文本关键词绕过。
- retired stage naming 不得通过文件名、路径名、Python symbol 或静态字符串拼接复活；合法版本面使用 `QRP 0.1`、`0.1`、`0.1.0` 和 `/api/0.1`，不使用带 `v` 前缀的 `0.1` API 版本路径。

## 验证原则

所有验证结果必须使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`。`PASS` 必须对应实际命令或可检查 artifact。未运行、阻塞或超时的检查不能写成 PASS。

常用治理验证：

```bash
git diff --check
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
python -m pytest tests/agents tests/architecture -q
```

分层测试入口：

```bash
python -m pytest tests/architecture/test_public_safe_evidence.py -q
python -m pytest tests/architecture/test_public_safe_evidence_data_contract.py -q -m data_dependency
python -m pytest tests/agents/handoff tests/architecture/test_handoff_package_contract.py -q -m "not slow_integration"
python -m pytest tests/agents/handoff/test_build_handoff_package_integration.py -q -m slow_integration
```

## 数据安全

禁止提交 private data、vendor marker、本地绝对路径、股票代码样本、row-level sample 或 raw traceback。REVIEW_SAFE evidence 只能包含摘要、schema、计数、hash、命令状态和路径结构；run output 架构主轴是 LOCAL_ONLY / REVIEW_SAFE，不是 private / public。
