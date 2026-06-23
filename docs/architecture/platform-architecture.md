# Platform Architecture

QRP 采用分层研究平台结构。本文只描述长期架构，不记录当前阶段。

## 分层

```text
sources  -> raw/interim source ingestion
storage  -> datasets, manifests, fingerprints, comparisons
contracts -> fields, schemas, ids, time, privacy, workflows
research -> factors, features, labels, models, predictions, evaluations, portfolios, signals
compute  -> dataframe/tensor math and portfolio helpers
runtime  -> resources, logging, privacy, metrics
reports  -> public-safe evidence and report artifacts
interfaces -> API, CLI, services
agents   -> workflow planning and proposals
```

## 数据流

```text
source data -> dataset -> factor -> feature -> model -> prediction -> signal -> evaluation -> portfolio -> report/evidence
```

signal 可以不经过 model，直接来自规则或 factor。model 输入应通过 feature contract 明确，不应把 raw factor 和 model feature 混写。

## Evidence 流

runtime output 和 handoff output 写到 `.run/` 或用户指定输出目录。tracked docs 不保存历史 evidence。

## 扩展原则

新增层级或 public surface 前必须说明：

- owner layer
- input contract
- output contract
- validation gate
- private safety impact
- migration/removal plan
