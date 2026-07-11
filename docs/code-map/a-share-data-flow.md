# A-share Data Flow

本文描述 A-share 日频研究数据流和边界，不暴露具体内部数据供应商。

## 数据路径

```text
internal-provider raw exports -> source contracts -> interim normalization -> dataset artifacts -> research datasets -> factors/features/labels -> evaluation/portfolio/report
```

当前 A-share 日频默认边界保持为 `2014-01-01` 到 `2025-12-31`；
调用方显式传入 start/end date 时仍以调用方窗口为准。repo-root 数据链路使用
统一 storage layout：

```text
internal-provider raw root
-> internal-provider interim dataset root
-> data/processed/a_share
-> data/research/a_share
-> evaluation output
```

该链路描述长期目录语义，不表示当前工作树中某次运行状态。

## sources 层

sources 层负责内部供应商 raw schema、字段映射、raw/interim 转换和内部输入标识。它不承载 A 股市场语义、研究算法或对外数据源选择面。
source implementation owns provider raw-to-interim normalization：供应商股票代码和
指数代码在 source 层 fail closed，不反向依赖 A-share market owner。

## markets 层

`markets/` 是通用市场领域层，不绑定 A 股。通用 session window 位于 `markets/session_windows.py`。`markets/implementations/a_share/` 负责 A 股 calendar、processed data、mask、factor/label binding、profile 和 scenario。
`markets/implementations/a_share/validation/strict_values.py` owns A-share
semantic strict values，包括 stock source asset id、index source asset id、
market type、trading-day bool 和 A-share owner-local timestamp validation。

A-share active processed profile 仍为日频。minute/tick 只由通用 data frequency contract 表达，A-share processed materialization 对 active profile 以日频 fail-closed。mask owner 是 `markets/implementations/a_share/processed/annotations.py`；research/factors、research/labels 和 research/evaluations 不重算 A-share mask。

`markets/implementations/a_share/processed/materializer.py` 保留 `materialize_processed_data()` 入口和窄 orchestration。内部输入映射由 processed owner-local mapping module 负责；dependency graph、profile frequency、input contract 和 required column validation 位于 `materialization_validation.py`；dataset root、processed input read、chunk iteration 和 write/inspect 位于 `materialization_io.py`。panel、annotation、factor 和 label materialization 继续通过当前 processed owner modules 调用 panel compute、`annotations.derive_annotations()`、factor definitions 和 label definitions。

processed owner 内部跨文件生产 helper 使用非下划线 owner-local 名称，并且不从 `processed/__init__.py` 重导出。下划线 helper 只表示同文件私有实现细节。

A-share processed output IO owns the resource-policy disk gate for its staged
dataset lifecycle. It checks same-volume free space before scratch creation,
between produced chunks, and before staged publication; storage remains unaware
of `RuntimeResourcePolicy`. The same owner derives physical targets for source,
processed, calendar, coverage, and sizing-probe reads from the active policy and
routes them through storage's single physical iterator. A minimum session that
exceeds the policy fails before panel/factor/label computation.
Sizing probes materialize one core session at a time, retain only the maximum
measurement, and pass the active policy as the total storage materialization
bound. The owner measurement and production write path both count retained
source inputs, the current physical/Pandas input, computed and filtered frames,
owner state, and four simultaneously possible selected-output copies. Their sum
must fit one `effective_chunk_bytes` budget. Trading-session discovery retains
only the requested lookback, core window, and lookahead while scanning bounded
calendar chunks; panel calculation preserves complete historical calendar
semantics while projecting only the fields required by the A-share owner.
Dataset-session discovery splits the caller budget between bounded physical
chunks and the deep-sized retained session namespace. Verification aggregates
likewise reserve separate physical-read and exact-key SQLite-cache allocations.

A-share coverage uses trading sessions. Coverage helpers must reject invalid
asset keys and timestamps instead of silently dropping them. Their cached
asset/session namespace is deep-sized incrementally and must fit its owner-local
state allocation instead of growing as an unbudgeted Python mapping. Crypto
coverage uses its own minute-grid semantics and must not reuse A-share calendar
rules.

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
