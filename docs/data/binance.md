# Binance 数据合同

本文描述 Binance 公开市场数据在 QRP 中的当前数据合同。本文只记录结构和层边界，不记录真实 payload、行级样本、运行结果或本地路径。

## Canonical 本地 Envelope

Canonical local-envelope 导入器只接受 JSONL。每行必须是 canonical envelope，所有
`data_kind` 共同必需的顶层字段为：

```text
source
market_type
data_kind
symbol
captured_at
payload
```

`source` 必须是 `binance`，`market_type` 必须是 `spot`。`captured_at`
必须是带 `Z` 或 `+/-HH:MM` UTC offset 的 ISO timestamp 字符串；numeric
timestamp 和没有 timezone offset 的字符串不是当前合同。当前 `data_kind` 只允许：

```text
klines
agg_trades
depth
```

条件字段：

- `klines`：`interval` 必填且必须为 `1m`。
- `agg_trades` / `depth`：`interval` 必须缺省。

严格标量规则：

- numeric/integer 字段拒绝 `bool`、`NaN`、`inf`、`-inf` 和非数值。
- integer 字段拒绝小数值。
- boolean 字段只接受 JSON boolean，不接受字符串或 numeric alias。
- `captured_at`、kline `open_time`、kline `close_time` 和 aggregate trade `timestamp` / `T` 使用共享的 Binance strict timestamp validators。`captured_at` 必须是带 UTC offset 的 ISO timestamp；market payload 时间必须是非负毫秒时间戳。
- `1m` kline 的 `open_time` 必须整分钟对齐，`close_time` 必须严格等于 `open_time + 59_999ms`。名义为 `1m` 但时间窗口不闭合的 payload 由 raw owner fail closed。
- aggregate trade `timestamp` / `T` 在 raw owner 校验，不把非法时间延迟到 interim normalization。

Binance market payload 的毫秒时间戳还必须落在 QRP 下游可安全表示的 pandas 纳秒时间戳范围内。上界由 `pd.Timestamp.max` 推导；epoch zero、上界内的值返回 UTC-aware 纳秒 `Timestamp`，首个越界值和更大的整数都由 Binance source owner fail closed。Raw kline、raw aggregate trade 与 interim normalization 复用同一范围合同。

`klines` 的 `payload` 是 connector 返回的 kline array，允许 connector 追加额外字段；当前语义校验只读取已批准字段。`agg_trades` 和 `depth` 的 `payload` 是 connector 返回的 object。裸 payload、非 envelope JSON、非 JSONL 文件和不支持的 data kind 都不是当前合同。

Raw envelope structure 不因语义校验而改变。当前 raw validation 会检查 kline 时间窗口、OHLC、volume、trade count、aggregate trade timestamp/price/quantity/id range，以及 depth price/quantity。Kline semantic validation 还要求 `taker_buy_volume <= volume`、`taker_buy_quote_volume <= quote_volume`，numeric payload 拒绝 `NaN`、`inf`、`-inf` 和非数值。双边 best bid/ask 同时存在时必须满足 bid < ask；单边 depth payload 可保留为 raw valid，后续 panel 层用 quote observation diagnostics 表达不可用 quote。

Depth normalization 在 raw 和 interim 路径复用同一个 owner helper。`bids` 必须按 price 严格递减排列，`asks` 必须按 price 严格递增排列，每个 raw level 必须恰好是 `[price, quantity]`，level 1 表示校验后的 best bid 或 best ask。同一 asset、timestamp、side 下的重复 depth level 在 market-panel gate 中 fail closed。

## Raw 布局

Raw 文件从以下位置发现：

```text
data/raw/binance/spot/klines/<symbol>/1m/*.jsonl
data/raw/binance/spot/agg_trades/<symbol>/*.jsonl
data/raw/binance/spot/depth/<symbol>/*.jsonl
```

这些路径仅是 canonical local-envelope 的输入布局。它们不定义 public API、public CLI 或持久化研究输出。该 importer 路径不进行网络采集。下面独立的 recorder-run 合同只使用未认证的 Binance public market-data endpoints，且从不读取 API keys 或 secrets。

## Recorder Run 合同

每个 recorder root 包含一个 `run.json`，以及该 manifest 声明的 gzip JSONL members，不会通过 glob 合并多个 run。`recording_scope` 为每个 output root 选择一个独立的当前合同：

```text
futures-hft:
  USDⓈ-M Futures diff depth @ 100ms
  USDⓈ-M Futures aggregate trades
  -> consumed by the current HFT workflow

spot-l2:
  Spot diff depth @ 100ms
  -> independent long-lived L2 capture for future spot/perpetual research
```

两个 scope 使用独立的 root、manifest 和 aggregate status。Spot 中断不能降级或拒绝完整的 Futures HFT run。现有 Spot replay、canonical local-envelope 输入和 minute Crypto research 仍是相互独立的能力。每条命令都是一个托管的数据规模操作，并在其生命周期内持有 host-local QRP lock，因此同一主机上不会并发记录 `futures-hft` 和 `spot-l2`。请顺序运行，或使用不同主机。提供的 output root 必须不存在；recorder 成功时以原子方式发布。

```text
python -m quant_research_platform.sources.binance.recorder --scope futures-hft --output-root <run-root> --duration-seconds <seconds>

python -m quant_research_platform.sources.binance.recorder --scope spot-l2 --output-root <run-root> --duration-seconds <seconds>
```

记录从命令启动时开始；未捕获的时段无法回填完整 L2。

RPI depth、book ticker、force order、mark price 和辅助统计不属于当前 recorder 合同，因为已实现的研究链不消费它们。

每条 raw 记录包含：

```text
capture_seq
receive_time
receive_monotonic_ns
connection_id
segment_id       # depth and snapshot only
payload          # lossless exchange payload
```

`capture_seq` 只有一个 owner：串行化的 run writer。每个 manifest member 都由 exact path、market、stream、data kind、row count、capture-sequence range、maximum uncompressed record bytes、compressed bytes 和 SHA-256 绑定。Replay 在使用前后校验 manifest、exact member set、hashes 和 physical identities。缺失的 capture sequence 永不回退到时间戳顺序。

Depth reconnect 会创建新的 `connection_id` 和 `segment_id`。匹配的 REST snapshot 绑定到相同 identity；replay 在有界本地 scratch 上建立 snapshot 索引，从不把 reconnect segment 当作连续数据。Spot 从 `lastUpdateId + 1` 启动；Futures 从 `lastUpdateId` 启动，随后要求 `pu == previous u`。数量为零会删除一个 level。Price identity 使用整数 ticks，而不是 binary-float equality。gap、crossed book、单边 book 或已知 depth 不足都会产生明确的 invalid boundary；不会静默复用过期 book。

以下时钟在整个 PANEL lineage 中保持独立：

```text
exchange event time (E)
Futures transaction time (T)
local receive UTC time
local monotonic receive time
capture sequence
segment identity
```

已实现的 HFT 数据链为：

```text
one recorder run
-> deterministic Futures L2 events
-> persisted 100ms PANEL sufficient statistics
-> FACTOR and LABEL derived only from persisted PANEL
-> standard ResearchDataset
-> standard evaluation / evaluation artifact
-> temporal train/test ModelSpec fitting and persisted predictions
-> out-of-sample prediction/target evaluation for every model
-> price-model taker economics and baseline-versus-score-gated maker economics
-> execution artifact and report
```

价格分支在同一个 500 ms return target 上比较 linear regression、LightGBM 和 XGBoost。execution-quality 分支预测明确命名的
`maker_bid_sell_pressure_proxy_500ms`：未来全部 observed sell-taker volume 除以当前 displayed best-bid quantity，并裁剪到 `[0, 1]`。分子包含在未来每个价格观察到的 sell-taker volume；PANEL 不提供按价格分布的 trade volume，因此这不声称交易发生在当前 bid。它在 prediction score 为 `0.5` 时 gate maker quote schedule，并将该 run 与未 gate 的 maker baseline 并列报告。这个 public-L2 proxy 不是 fill probability、MBO queue position，也不承诺可执行 fills。

recorder 使用有界的 `websockets` runtime 连接 public streams，使用 `httpx` 获取 public REST metadata/snapshots。它没有 authenticated endpoint 或 order API。

## Interim 数据集

`sources/binance/interim.py` 负责 raw envelope 到 interim normalization：

Raw 文件及连续的 normalization batch 必须已经按注册的 interim primary key 严格有序。Normalization 保留该物理顺序，并在文件与 batch 之间携带一个 last-key watermark；不会通过排序 batch 来掩盖回退的依赖。因而 aggregate trades 按 `(source_asset_id, timestamp, aggregate_trade_id)` 前进，depth levels 按 `(source_asset_id, timestamp, side, level)` 前进，bars 按 `(source_asset_id, timestamp)` 前进。重复或回退的行对每个 batch size 都以相同结果 fail closed。Depth 在每个 snapshot 内按注册的 lexical side order 输出，而 raw bid/ask price ordering 仍由上面的 source semantic checks 管理。

```text
klines      -> market_bars_1m
agg_trades  -> aggregate_trades_tick
depth       -> depth_levels_tick
```

Interim materialization 以有界 batch 读取 raw envelopes，在 normalization 前校验 `data_id` 和 raw `data_kind`，对 source timestamps 应用请求的 start/end 过滤，并通过 `storage.write_dataset` 写入 source-normalized datasets。`sources/` 只拥有 provider field mapping；市场语义和 research factors 由 `markets/implementations/crypto/` 与 `research/` 持有。

当 raw 文件存在但请求的过滤移除所有行时，interim materialization 仍会为该 `data_id` 写入 empty schema dataset。在窄时间窗口中，aggregate trades 或 depth 等可选 feed 的 empty schema dataset 是有效的；没有发现 raw 文件的 `data_id` 仍会跳过。

## 安全边界

Binance public market data 在转换为 REVIEW_SAFE 摘要或生成测试 fixture 前，仍按本地输入处理。不要将 raw connector payload 复制到 docs、reports、最终答复或 REVIEW_SAFE 证据中。
