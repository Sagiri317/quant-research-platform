# Platform Flow

本文描述 QRP 当前代码流程。QRP 是 `local research platform`，只覆盖本地研究、
验证和 public-safe evidence；边界是 `no trading execution` 和
`no production trading boundary`。

## 总体链路

```text
sources -> storage -> contracts -> research -> compute -> reports -> interfaces
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
tests 和 handoff evidence 只保护当前合同，不把历史阶段、旧路径或旧字段当成
当前事实。

## 关键目录

```text
src/quant_research_platform/sources
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

contracts 层定义字段、ID、时间、privacy、data、factor、feature、label、model、prediction、signal、evaluation、portfolio、report 和 workflow 的当前语义。

## storage

storage 层处理 dataset layout、atomic write、checksums、dataset read/write、fingerprint 和 compare。planning 不应依赖昂贵内容级 fingerprint。

## research

research 层实现因子、特征、标签、模型、预测、评估、组合和 signal materialization。

## compute

compute 层提供 dataframe/tensor 公共计算能力。重复逻辑应收敛，不应在业务层复制。

## reports

reports 层只生成 public-safe evidence 和 report artifacts，不保存历史状态事实。
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
`.agents tooling` 是 tracked governance tooling source；`.run output/evidence`
只是运行输出和 evidence layer，不定义架构、工具入口或当前状态。
