# A-share Data Flow

本文描述 A-share 日频研究数据流和边界，不暴露具体内部数据供应商。

## 数据路径

```text
internal-provider raw exports -> source contracts -> interim normalization -> dataset artifacts -> research datasets -> factors/features/labels -> evaluation/portfolio/report
```

## sources 层

sources 层负责内部供应商 raw schema、字段映射、raw/interim 转换和 source-local ID。它不承载 A 股市场语义、研究算法或对外 provider selector。

## markets 层

`markets/` 是通用市场领域层，不绑定 A 股。通用 session window 位于 `markets/session_windows.py`。`markets/implementations/a_share/` 负责 A 股 calendar、processed data、mask、factor/label binding、profile 和 scenario。

A-share active processed profile 仍为日频。minute/tick 只由通用 data frequency contract 表达，A-share processed materialization 对 active profile 以日频 fail-closed。mask owner 是 `markets/implementations/a_share/processed/annotations.py`；research/factors、research/labels 和 research/evaluations 不重算 A-share mask。

`markets/implementations/a_share/processed/materializer.py` 保留 `materialize_processed_data()` 入口和窄 orchestration。source ID mapping 位于 `source_ids.py`；dependency graph、profile frequency、input contract 和 required column validation 位于 `materialization_validation.py`；dataset root、processed input read、chunk iteration 和 write/inspect 位于 `materialization_io.py`。panel、annotation、factor 和 label materialization 继续通过当前 processed owner modules 调用 panel compute、`annotations.derive_annotations()`、factor definitions 和 label definitions。

processed owner 内部跨文件生产 helper 使用非下划线 owner-local 名称，并且不从 `processed/__init__.py` 重导出。下划线 helper 只表示同文件私有实现细节。

## storage 层

storage 层负责 dataset artifact 的读写、manifest、schema、partition、fingerprint 和 compare。private data 只能在本地 workspace 中使用，不进入 repo。

## research 层

research 层只实现通用研究算法。涨跌停、ST、停牌、A 股日历等市场语义由 A 股实现层绑定，research 不直接 import sources 或具体市场实现。

## REVIEW_SAFE evidence

A-share 验证报告只能公开：

- row count 摘要
- schema hash
- partition summary
- command status
- artifact hash
- REVIEW_SAFE issue summary

禁止公开：

- raw row-level data
- vendor marker
- 本地绝对路径
- stock-code sample
- private traceback
