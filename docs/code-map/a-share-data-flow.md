# A-share Data Flow

本文描述 A-share 日频研究数据流和边界，不暴露具体内部数据供应商。

## 数据路径

```text
internal-provider raw exports -> source contracts -> interim normalization -> dataset artifacts -> research datasets -> factors/features/labels -> evaluation/portfolio/report
```

## sources 层

sources 层负责内部供应商字段映射、raw schema、raw export 描述、interim 转换和 source catalog。它不应把 private row-level sample 写入 tracked docs 或 public evidence。

## storage 层

storage 层负责 dataset artifact 的读写、manifest、schema、partition、fingerprint 和 compare。private data 只能在本地 workspace 中使用，不进入 repo。

## research 层

research 层使用 dataset contract 构造 factor、feature、label 和 evaluation 输入。涨跌停、ST、停牌、市场日历等 A-share 边界应通过 contracts 和 research pipeline 明确，不应散落在文档叙事中。

## public-safe evidence

A-share 验证报告只能公开：

- row count 摘要
- schema hash
- partition summary
- command status
- artifact hash
- public-safe issue summary

禁止公开：

- raw row-level data
- vendor marker
- 本地绝对路径
- stock-code sample
- private traceback
