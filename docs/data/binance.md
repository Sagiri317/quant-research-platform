# Binance Data Contract

本文描述 Binance public market data 在 QRP 中的当前数据合同。本文只记录结构和层边界，不记录真实 payload、row-level sample、运行结果或本地路径。

## Raw Envelope

Binance raw input 只接受 JSONL。每行必须是 canonical envelope，顶层字段为：

```text
source
market_type
data_kind
symbol
captured_at
payload
interval
```

`source` 必须是 `binance`，`market_type` 必须是 `spot`。当前 `data_kind` 只允许：

```text
klines
agg_trades
depth
```

`klines` 必须声明 `interval=1m`，`payload` 是 connector 返回的 kline array。`agg_trades` 和 `depth` 的 `payload` 是 connector 返回的 object。裸 payload、非 envelope JSON、非 JSONL 文件和 unsupported data kind 都不是当前合同。

## Raw Layout

Raw files are discovered from:

```text
data/raw/binance/spot/klines/<symbol>/1m/*.jsonl
data/raw/binance/spot/agg_trades/<symbol>/*.jsonl
data/raw/binance/spot/depth/<symbol>/*.jsonl
```

These paths are local input layout only. They do not define public API, public CLI, or durable research output.

## Interim Datasets

`sources/binance/interim.py` owns raw envelope to interim normalization:

```text
klines      -> market_bars_1m
agg_trades  -> aggregate_trades_tick
depth       -> depth_levels_tick
```

Interim materialization reads raw envelopes in bounded batches and writes source-normalized datasets through `storage.write_dataset`. `sources/` owns provider field mapping only; market semantics and research factors are owned by `markets/implementations/crypto/` and `research/`.

## Safety Boundary

Binance public market data is still treated as local input until transformed into REVIEW_SAFE summaries or generated test fixtures. Do not copy raw connector payloads into docs, handoff, reports, final answers, or REVIEW_SAFE evidence.
