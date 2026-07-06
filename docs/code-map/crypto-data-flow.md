# Crypto Data Flow

本文描述当前 crypto Binance 研究链路的 owner-flow。本文不记录运行状态、branch、commit、真实 symbol 样本、raw payload 或本地路径。

## Owner Flow

```text
sources/binance/raw_exports.py
-> sources/binance/interim.py
-> markets/implementations/crypto/processed/
-> research/factors/microstructure/
-> research/labels/forward_returns.py
-> markets/implementations/crypto/research_profiles.py
-> markets/implementations/crypto/research_scenarios.py
```

`sources/binance/` owns raw envelope validation, file discovery, and interim normalization. It does not own market alpha semantics.

`markets/implementations/crypto/processed/` owns crypto processed panel, microstructure factor dataset, and forward-return label dataset materialization. Processed materializers read upstream datasets through `DatasetReadMode.CHUNKED` and use `RuntimeResourcePolicy` to keep the control path chunk-aware.

`research/factors/microstructure/` owns generic dataframe calculations for the approved microstructure factors. It does not write files.

`research/labels/forward_returns.py` owns generic forward-return label calculation. It does not write files.

## Current Datasets

```text
market_bars_1m
aggregate_trades_tick
depth_levels_tick
market_panel_1m
microstructure
forward_returns
microstructure_forward_returns
```

`market_panel_1m` joins minute bars with aggregate trade flow and order-book snapshots. Missing aggregate trades for a minute produce zero signed volume. Missing depth snapshots leave depth-derived fields null.

## Current Evaluation Signals

The crypto evaluation scenario uses the approved microstructure factor set:

```text
spread
relative_spread
depth_imbalance
microprice
microprice_deviation
trade_intensity_5m
signed_volume_5m
flow_imbalance_5m
```

Lower spread signals are interpreted as `lower_is_better`; the other listed signals use `higher_is_better`.

## Current Labels

The crypto research dataset uses only the approved forward-return labels:

```text
forward_return_1m
forward_return_5m
forward_return_15m
```

## Boundary

The crypto flow is a local research flow. It does not add execution connectors, exchange execution, production trading, external LLM calls, public API endpoints, public CLI commands, scheduler, worker, or deployment surface.
