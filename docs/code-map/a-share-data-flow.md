# A-share Data Flow

本文只描述 A-share 日频研究链路的长期 owner flow、资源生命周期与公开边界，不记录某次运行状态或 private provider 细节。

## Owner Flow

```text
source raw exports -> source interim normalization
-> markets/implementations/a_share/processed
-> storage datasets
-> research/evaluation/portfolio
-> reports -> artifacts
```

- `sources/` 负责 raw schema、field mapping、source-local ID 与 raw/interim normalization，不承载 A-share 市场语义。
- `markets/session_windows.py` 是通用 session window owner；`markets/implementations/a_share/` 负责 calendar、mask、processed binding、profile 与 scenario。
- processed materializer 只编排 owner-local planning、validation、IO 与 panel/annotation/factor/label callbacks。
- `storage/` 负责 dataset physical IO、manifest、schema、exact keys、fingerprint、compare 与 publication。
- `research/` 只负责通用计算；涨跌停、ST、停牌和交易日等 market semantics 留在 A-share implementation。

## Lifecycle Invariant

- raw/interim traversal 在读取或 decode 前做 owner admission，并在 normal、error 或 early close 时释放 scratch 与 iterator。
- processed flow 在一个 `RuntimeResourcePolicy` scope 内完成 session discovery、active-asset traversal、materialization 和 staged publication。
- active materialization 维持 deterministic asset/session order、跨 chunk primary-key 单调性和 owner-local bounded retained state。
- Storage-owned IO 使用同一 operation policy 管理自身 retained state，并独立执行同卷 disk headroom checks。
- A-share coverage 使用 trading sessions；Crypto minute-grid 由其 market owner 独立定义，不复用 A-share calendar。

## REVIEW_SAFE Boundary

公开 evidence 只包含摘要、计数、schema/hash 与命令状态；private、provider、row-level、absolute-path 和 raw traceback 内容保持 LOCAL_ONLY。
