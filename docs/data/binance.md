# Binance Data Contract

本文描述 Binance public market data 在 QRP 中的当前数据合同。本文只记录结构和层边界，不记录真实 payload、row-level sample、运行结果或本地路径。

## Raw Envelope

Binance raw input 只接受 JSONL。每行必须是 canonical envelope，所有
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

Conditional fields:

- `klines`: `interval` required and must be `1m`.
- `agg_trades` / `depth`: `interval` must be absent.

`klines` 的 `payload` 是 connector 返回的 kline array，允许 connector 追加额外字段；当前语义校验只读取已批准字段。`agg_trades` 和 `depth` 的 `payload` 是 connector 返回的 object。裸 payload、非 envelope JSON、非 JSONL 文件和 unsupported data kind 都不是当前合同。

Raw envelope structure 不因语义校验而改变。当前 raw validation 会检查 kline OHLC、volume、trade count、aggregate trade price/quantity/id range，以及 depth price/quantity。Kline semantic validation 还要求 `taker_buy_volume <= volume`、`taker_buy_quote_volume <= quote_volume`，numeric payload 拒绝 `NaN`、`inf`、`-inf` 和 nonnumeric values。双边 best bid/ask 同时存在时必须满足 bid < ask；one-sided depth payload 可保留为 raw valid，后续 panel 层用 quote observation diagnostics 表达不可用 quote。

## Raw Layout

Raw files are discovered from:

```text
data/raw/binance/spot/klines/<symbol>/1m/*.jsonl
data/raw/binance/spot/agg_trades/<symbol>/*.jsonl
data/raw/binance/spot/depth/<symbol>/*.jsonl
```

These paths are local input layout only. They do not define public API, public CLI, or durable research output.
`sources/binance` runtime 只读取这些本地 JSONL 文件，不联网、不读取 API key
或 secret，也不把 connector 调用逻辑写入 repo runtime。

## Interim Datasets

`sources/binance/interim.py` owns raw envelope to interim normalization:

```text
klines      -> market_bars_1m
agg_trades  -> aggregate_trades_tick
depth       -> depth_levels_tick
```

Interim materialization reads raw envelopes in bounded batches, validates `data_id` and raw `data_kind` before normalization, applies requested start/end filtering to source timestamps, and writes source-normalized datasets through `storage.write_dataset`. `sources/` owns provider field mapping only; market semantics and research factors are owned by `markets/implementations/crypto/` and `research/`.

## Safety Boundary

Binance public market data is still treated as local input until transformed into REVIEW_SAFE summaries or generated test fixtures. Do not copy raw connector payloads into docs, handoff, reports, final answers, or REVIEW_SAFE evidence.
