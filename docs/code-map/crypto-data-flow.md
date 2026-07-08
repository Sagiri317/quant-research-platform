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

`markets/implementations/crypto/processed/` owns crypto market binding and processed semantics. Processed source-input selector 为当前 crypto profile 选择必需的 source-normalized interim input data ids；Binance 是 `crypto_binance_1m` 中当前唯一注册的 source owner。Source-input bridge 要求 expected input ids 精确匹配，missing、duplicate 或 extra ids 都 fail closed。`processed/panels.py` owns pure crypto market-panel calculation, and `processed/materialize_panel.py` only orchestrates IO/chunking. Processed materializers read upstream datasets through `DatasetReadMode.CHUNKED` and use `RuntimeResourcePolicy` to keep the control path chunk-aware.

`research/factors/microstructure/` owns generic microstructure factor definitions and dataframe calculations. It does not import crypto market implementation modules and does not write files.

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

`market_panel_1m` joins minute bars with aggregate trade flow and order-book snapshots. `asset_id` is the canonical crypto asset id; source-local asset identifiers stay in Binance interim data and do not enter processed or research schemas.

Trade-flow and quote observation diagnostics are explicit context columns:

```text
trade_flow_observed
quote_observed
quote_age_seconds
```

`trade_flow_observed` distinguishes real zero trade flow from missing aggregate trade feed. A zero-trade bar with no matching aggregate trades can produce `signed_volume=0` and `trade_flow_observed=True`; a positive-trade bar without matching aggregate trades keeps `signed_volume` null and `trade_flow_observed=False`.

Quotes are selected as-of bar `close_time`, never after bar close. `quote_observed=True` only when the as-of quote exists and `quote_age_seconds` is within the approved max quote age. Missing or stale quotes leave quote-derived fields null and `quote_observed=False`.

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
