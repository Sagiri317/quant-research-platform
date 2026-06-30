# Platform Architecture

QRP 采用分层研究平台结构。本文只描述长期架构，不记录当前阶段。

## 分层

```text
policy     -> 共享安全规则、ID 安全判断、状态聚合
contracts  -> 结构合同、枚举、ID、TimeWindow、基础校验
catalog    -> generic registry class
profiles   -> API/CLI/agent/workflow 共用组合根
sources    -> provider/source 摄取、raw schema、字段映射、raw/interim 转换
markets    -> 通用市场领域、session window、市场实现容器
markets/implementations/a_share -> A 股 calendar、processed、mask、研究绑定
research   -> 通用研究计算，只构建 result
compute    -> dataframe/tensor、portfolio math、performance 数值核心
reports    -> 只 build/render 已准备好的 evidence/report payload
artifacts  -> 写 LOCAL_ONLY outputs 和 REVIEW_SAFE evidence 文件
workflows  -> research -> diagnostics payload -> reports -> artifacts 编排
storage    -> dataset IO、manifest、fingerprint、compare、data layout
runtime    -> resources、logging、workflow records、metrics payload
interfaces -> 已批准的 API、CLI、service adapter
agents     -> proposal-only agent
.agents    -> governance、handoff、prompt tooling
```

## 数据流

```text
source data -> market implementation -> storage dataset -> research result -> workflows diagnostics payload -> reports payload -> artifacts
```

signal 可以不经过 model，直接来自规则或 factor。model 输入应通过 feature contract 明确，不应把 raw factor 和 model feature 混写。

## Evidence 流

LOCAL_ONLY outputs 写到用户指定或 workflow 生成的 `<run_root>`；REVIEW_SAFE handoff evidence 写入 handoff package。

`run.json` 是唯一运行级机器事实源，`summary.md` 是唯一运行级人读摘要，实例结果位于 `outputs/<output_id>/`。所有 QRP-owned JSON 使用不带版本后缀的 `schema_id`。

`research/` 可以使用 contracts、storage、runtime resources 和 `markets.session_windows`，但不得 import 具体市场实现或 sources。`runtime/` 不 import reports，也不写 report/storage artifact。`workflows/diagnostics/` 可以读取 storage dataset，并把 schema、count、hash、status 等 REVIEW_SAFE payload 传给 reports。`reports/` 只 build/render 已准备好的内容，不读取 storage dataset，不写文件，不重算 research、evaluation 或 portfolio 逻辑。`artifacts/` 只写文件，不 build evidence payload，不 import research、reports 或具体市场实现。

## 扩展原则

新增层级或 public surface 前必须说明：

- owner layer
- input contract
- output contract
- validation gate
- private safety impact
- 收敛计划
