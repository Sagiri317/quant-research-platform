# Platform Flow

本文描述 QRP 当前代码流程。QRP 是 `local research platform`，只覆盖本地研究、
验证和 public-safe evidence；边界是 `no trading execution` 和
`no production trading boundary`。

## 总体链路

```text
sources -> markets implementation -> storage -> research -> compute -> reports
```

研究链路：

```text
dataset -> factor -> feature -> label -> model -> prediction -> signal -> evaluation -> portfolio -> evidence
```

长期合同锚点：

```text
data -> feature -> signal/model -> evaluate -> evidence
```

该链路受 `current contract` 和 `current-contract-only` 约束：文档、脚本、
tests 和 handoff evidence 只保护当前合同。

## 关键目录

```text
src/quant_research_platform/sources
src/quant_research_platform/markets
src/quant_research_platform/markets/implementations/a_share
src/quant_research_platform/storage
src/quant_research_platform/contracts
src/quant_research_platform/research
src/quant_research_platform/compute
src/quant_research_platform/runtime
src/quant_research_platform/reports
src/quant_research_platform/interfaces
src/quant_research_platform/agents
```

## contracts

contracts 层定义结构合同、枚举、ID、TimeWindow 和基础校验，不放 pandas 算法、source/market 具体实现或模型算法事实源。

## sources 与 markets

sources 层只负责 provider/source ingestion、raw schema、字段映射和 raw/interim 转换。markets 是通用市场领域层，`markets/session_windows.py` 持有通用 session window；A 股 calendar、processed、mask 和研究绑定位于 `markets/implementations/a_share/`。

## storage

storage 层处理 dataset layout、atomic write、checksums、dataset read/write、fingerprint 和 compare。planning 不应依赖昂贵内容级 fingerprint。

## research

research 层实现通用因子、特征、标签、模型、预测、评估、组合和 signal materialization，不 import 具体市场实现或 sources。

## compute

compute 层提供 dataframe/tensor、portfolio math 和 performance 数值核心，不持有 market session domain 工具。

## runtime 与 workflows

runtime 提供 privacy、resources、logging、workflow records 和 execution metrics payload，不写 report artifact。workflows 只编排 workflow node 和 service 调用，不成为算法事实源。

## reports

reports 层只生成 public-safe evidence 和 report artifacts。
`private data safety` 是 reports 和 handoff 的硬边界。

## interfaces

interfaces 层提供 API、CLI、service 入口。新增 public surface 必须经用户批准。

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

## agents

agents 层提供 proposal 和 workflow planning，不得绕过 contracts 或 private safety。
`.agents tooling` 只承载 governance、handoff 和 prompt tooling；`.run output/evidence`
只是运行输出和 evidence layer，不定义架构、工具入口或当前状态。
