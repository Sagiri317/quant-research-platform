# Platform Architecture

QRP 采用分层研究平台结构。本文只描述长期架构，不记录当前阶段。

## 分层

```text
contracts  -> 结构合同、枚举、ID、TimeWindow、基础校验
catalog    -> 注册描述与组合入口
sources    -> provider/source 摄取、raw schema、字段映射、raw/interim 转换
markets    -> 通用市场领域、session window、市场实现容器
markets/implementations/a_share -> A 股 calendar、processed、mask、研究绑定
research   -> 通用研究算法
compute    -> dataframe/tensor、portfolio math、performance 数值核心
storage    -> dataset IO、manifest、fingerprint、compare、layout
runtime    -> privacy、resources、logging、workflow records、metrics payload
reports    -> public-safe report/evidence artifact
workflows  -> workflow node 与 service 编排
interfaces -> 已批准的 API、CLI、service adapter
agents     -> proposal-only agent
.agents    -> governance、handoff、prompt tooling
```

## 数据流

```text
source data -> dataset -> factor -> feature -> model -> prediction -> signal -> evaluation -> portfolio -> report/evidence
```

signal 可以不经过 model，直接来自规则或 factor。model 输入应通过 feature contract 明确，不应把 raw factor 和 model feature 混写。

## Evidence 流

runtime output 和 handoff output 写到 `.run/` 或用户指定输出目录。tracked docs 不保存历史 evidence。

`research/` 可以使用 contracts、storage、runtime resources 和 `markets.session_windows`，但不得 import 具体市场实现或 sources。`runtime/` 不 import reports，也不写 report/storage artifact。`reports/` 只负责 artifact 写出，不重算 research、evaluation 或 portfolio 逻辑。

## 扩展原则

新增层级或 public surface 前必须说明：

- owner layer
- input contract
- output contract
- validation gate
- private safety impact
- migration/removal plan
