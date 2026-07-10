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

The crypto research profile imports Binance interim data across the processed
dependency window required by factor warmup and label lookahead. The dependency
window remains owned by `markets/implementations/crypto/processed/`;
`sources/binance/` still owns only raw/interim normalization and does not own
factor or label semantics.

`markets/implementations/crypto/processed/` owns crypto market binding and processed semantics. Processed source-input selector 为当前 crypto profile 选择必需的 source-normalized interim input data ids；Binance 是 `crypto_binance_1m` 中当前唯一注册的 source owner。Source-input bridge 要求 expected input ids 精确匹配，missing、duplicate 或 extra ids 都 fail closed。`processed/panels.py` owns pure crypto market-panel calculation, and `processed/materialize_panel.py` only orchestrates IO/chunking. Processed materializers read upstream datasets through `DatasetReadMode.CHUNKED` and use `RuntimeResourcePolicy` to keep the control path chunk-aware.

`processed/materialization_validation.py` owns crypto materialization lifecycle validation. Every panel records immutable identities for its direct interim inputs, and every factor or label records the identity of its direct panel input. Each identity contains only the input dataset id plus hashes of `spec.json` and `_manifest/manifest.json`; it does not read row-level content or call `fingerprint_dataset`. A newly written dependency and a reused dependency pass through the same identity and exact-key validation.

Dependency coverage compares the complete registered primary key. Panel expected keys come from current source bars after canonical crypto asset and market binding; factor and label expected keys come from panel rows in the requested core window. Missing assets, missing intermediate minutes, extra keys, duplicates, or any `market_id` / `asset_id` / `timestamp` mismatch fail closed. Warmup and lookahead rows remain read-only dependencies and are not emitted as core output.

Factor and label streaming readers maintain a timestamp watermark per `(market_id, asset_id)`. Rows may be stably sorted inside one storage chunk, but a later chunk must begin strictly after that key's previous watermark. Reversed, overlapping, or regressing chunks fail before rolling-factor carry or forward-label buffering is updated.

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

Before any trade aggregation or quote as-of matching, the crypto market owner validates the normalized bar contract: `timestamp == open_time`, `timestamp` is minute-aligned, `close_time == timestamp + 59_999ms`, `interval == "1m"`, and `source_market_type == "spot"`. Non-empty bars, aggregate trades, and depth inputs must all carry the current crypto spot `source_market_type`; synthetic normalized sources pass through the same gate and are not assumed to have Binance raw provenance. An invalid bar close cannot extend matching into a later minute.

For every `(asset_id, timestamp, side)` depth snapshot, `level` starts at 1 and is contiguous. Bid prices strictly decrease with level and ask prices strictly increase. Duplicate levels, non-positive prices, negative quantities, and malformed sides fail closed. One-sided snapshots and zero-quantity levels remain structurally valid; quote observation still requires a valid two-sided, positive-size, uncrossed best quote.

Trade-flow and quote observation diagnostics are explicit context columns:

```text
trade_flow_observed
quote_observed
quote_age_seconds
```

`trade_flow_observed` distinguishes real zero trade flow from missing aggregate trade feed. A zero-trade bar with no matching aggregate trades can produce `signed_volume=0` and `trade_flow_observed=True`; a positive-trade bar without matching aggregate trades keeps `signed_volume` null and `trade_flow_observed=False`.

Quotes are selected as-of bar `close_time`, never after bar close. `quote_observed=True` only when the as-of quote timestamp is less than or equal to bar `close_time`, `quote_age_seconds <= DEFAULT_MAX_QUOTE_AGE`, `best_bid > 0`, `best_ask > 0`, `best_bid < best_ask`, `bid_quantity > 0`, and `ask_quantity > 0`. Missing, stale, crossed, one-sided, zero-size, or invalid quotes leave quote-derived fields null and `quote_observed=False`.

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
