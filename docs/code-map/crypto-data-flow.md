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

`markets/implementations/crypto/processed/` owns crypto market binding and processed semantics. Processed source-input selector 为当前 crypto profile 选择必需的 source-normalized interim input data ids；Binance 是 `crypto_binance_1m` 中当前唯一注册的 source owner。Source-input bridge 要求 expected input ids 精确匹配，missing、duplicate 或 extra ids 都 fail closed。`processed/panels.py` owns the production market-panel calculation from prepared bars plus trade/depth chunk iterables; tests exercise that same reducer path rather than a full-frame parallel calculator. Cross-file owner-local symbols use non-underscore names without package-root re-export. `processed/materialize_panel.py` only orchestrates IO/chunking. `DatasetReadMode.CHUNKED` 表达调用方的业务读取边界；实际 Parquet IO 复用 storage-owned physical batch iterator。The operation captures one `RuntimeResourcePolicy`, passes `payload_target_bytes` to physical reads and known frame-payload checks, and uses separate explicit disk reserves. Those checks are necessary payload bounds, not proof that reducer state or the complete process working set is bounded; the current implementation does not guarantee phase release, incremental flush or unlimited-depth spill.

Canonical asset ordering remains deterministic while the retained equality-group plan is live. Trade/depth outputs use the Runtime-owned Pandas payload diagnostic after materialization, and depth snapshot validation/finalization has one owner in `processed/panel_input_reducers.py`; `processed/panels.py` does not retain a parallel finalizer. The retained plan and reducer lifecycle are not represented as exact Python-state bytes.

`processed/materialization_validation.py` owns crypto materialization lifecycle validation. Every panel records immutable identities for its direct interim inputs, and every factor or label records the identity of its direct panel input. Each identity contains only the input dataset id plus hashes of `spec.json` and `_manifest/manifest.json`; it does not read row-level content or call `fingerprint_dataset`. A newly written panel, factor, or label completes identity and exact-key validation against its staged artifact before storage publishes the final root; failure removes the stage and leaves no final dataset. A reused dependency passes through the same validator against its already published artifact.

Dependency coverage compares the complete registered primary key through the storage-owned exact spillable key index. Expected and actual chunks are cast to the actual dataset's stored Arrow key schema and inserted into separate SQLite namespaces; duplicate keys fail closed, and indexed anti-existence queries check missing and extra keys exactly. Coverage is independent of physical row order, chunk size, and whether source-to-canonical asset mapping preserves order; it never constructs a full-dataset Python set, concat, global in-memory sort, or file-backed SQLite temp query. New-output coverage scratch is inside the staged dataset root and is removed before publication; validation of a reused published dependency uses same-volume sibling scratch. Both paths use each exact-key frame's actual Pandas payload for a same-volume reserve check before insertion and recover recognized PID-bound scratch after an ungraceful termination; disk reserve is independent from Runtime memory policy. Panel expected keys come from current source bars after canonical crypto asset and market binding; factor and label expected keys come from panel rows in the requested core window. Missing assets, missing intermediate minutes, extra keys, duplicates, or any `market_id` / `asset_id` / `timestamp` mismatch fail closed. Warmup and lookahead rows remain read-only dependencies and are not emitted as core output.

Factor and label computation still requires globally ordered `(market_id, asset_id, timestamp)` input and retains one last-key watermark; reversed, overlapping, completed-asset reappearance, or regressing chunks fail before state changes. The panel lifecycle writes one canonical asset per physical part and validates that invariant before publication and again before a dependent materializer can use its equality-group plan. Because physical storage is year-partitioned, factor and label materializers can then use the separately budgeted distinct asset namespace plus storage-owned equality and time predicates in canonical asset order without repeatedly decoding a mixed-asset part; a year boundary therefore cannot reintroduce an already completed asset, and physical decode work remains linear in input rows. Both materializers reuse one lifecycle-owned core-window mask. Factor calculation processes contiguous asset segments and retains only the active asset's fixed rolling carry, discarding it when the key advances, so prior assets are not reprocessed and carry state does not grow with asset history. Label calculation retains only the active asset's bounded lookahead buffer.

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

`trade_flow_observed` distinguishes a completeness-validated feed from a missing aggregate trade feed. Aggregate trade ids must be strictly increasing, underlying trade-id ranges must be contiguous and non-overlapping across both auxiliary-input chunks and physical bar-chunk boundaries, and per-bar absolute quantity plus covered trade count must match bar `volume` and `trade_count` under owner-local named tolerances. The panel lifecycle retains one compact aggregate-trade continuity watermark for the active source asset while per-bar reducer output is released. A zero-trade bar with no matching aggregate trades produces `signed_volume=0` and `trade_flow_observed=True`; a positive-trade bar without matching aggregate trades keeps `signed_volume` null and `trade_flow_observed=False`. Non-empty partial, duplicate, gapped, overlapping, or contradictory feed fails closed; merely seeing one trade never makes the feed observed.

Quotes are selected as-of bar `close_time`, never after bar close. `quote_observed=True` only when the as-of quote timestamp is less than or equal to bar `close_time`, `quote_age_seconds <= DEFAULT_MAX_QUOTE_AGE`, `best_bid > 0`, `best_ask > 0`, `best_bid < best_ask`, `bid_quantity > 0`, and `ask_quantity > 0`. Missing, stale, crossed, one-sided, zero-size, or invalid quotes leave quote-derived fields null and `quote_observed=False`.

Panel materialization enumerates the source-asset namespace, rejects canonical collisions, and processes assets in canonical `asset_id` order. Per-asset bars, trades, and depth use storage-owned equality and time predicates. After source-to-canonical binding, the panel orchestrator carries one strict `(market_id, asset_id, timestamp)` watermark across every bar chunk and asset; duplicates or regressions fail before reducer state changes, so exact set coverage cannot conceal invalid physical dependency order. Core windows read trades and depth through the storage physical iterator, and raw auxiliary rows are not concatenated across chunks or written to metadata or artifacts. Trade and depth reducers preserve continuity, duplicate, level, price, quantity, latest-invalid, future, exact and stale-quote semantics. The current resource checks do not claim that retained trade/depth candidates, reducer state, output rows, or the source-asset namespace are fully bounded. Phase release, bounded reducer state, incremental flush and unlimited-depth spill are not guarantees of the current implementation. Direct panel/factor/label writers apply a same-volume reserve check to each actual Pandas payload chunk before handing it to storage; they do not claim a separate fixed pre-scratch or pre-publication disk budget.

Quote matching uses asset-isolated, backward `merge_asof` at bar close with exact matches allowed and `DEFAULT_MAX_QUOTE_AGE` as tolerance. The latest snapshot is selected before `quote_valid` is applied, so an invalid latest quote never falls back to an older valid quote. Matching retains the original bar key and the final panel remains deterministically sorted by `(market_id, asset_id, timestamp)`.

Panel source-asset discovery preserves exact Unicode identity and rejects a source bar part assigned to more than one source asset before any per-asset reread, so equality filtering cannot turn one mixed-asset physical part into repeated decode work. It does not claim an exact Python mapping/string byte bound, and bounded namespace lifecycle is not a guarantee of the current implementation.

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
