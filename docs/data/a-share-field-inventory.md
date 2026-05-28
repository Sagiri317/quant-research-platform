# A 股字段可用性清单

本文记录当前 A 股字段在 full panel、slim factor table、evaluation input 和 N5 diagnostics 中的可用性。它是 source-neutral 的字段可用性索引，不替代代码 schema registry、pipeline 代码或测试。

## Source Of Truth

当前字段事实来自以下代码入口：

- CSMAR raw/interim 字段映射：`src/quant_research_platform/markets/implementations/a_share/csmar/contracts.py`
- Interim pruning 和 slim factor base columns：`src/quant_research_platform/markets/implementations/a_share/csmar/columns.py`
- Full panel 生成：`src/quant_research_platform/markets/implementations/a_share/datasets/build_panel.py`
- Slim factor table 生成：`src/quant_research_platform/markets/implementations/a_share/factors/build_factors.py`
- Price-volume factors 和 forward labels：`src/quant_research_platform/markets/implementations/a_share/factors/price_volume.py`
- A 股 evaluation adapter：`src/quant_research_platform/markets/implementations/a_share/evaluation/evaluation_adapter.py`
- N5 v1 diagnostics：`src/quant_research_platform/evaluations/diagnostics/`
- Evaluation artifact table contract：`src/quant_research_platform/artifacts/schema_registry.py`

## Status Taxonomy

| Status | Meaning |
|---|---|
| `available` | 当前链路生成该字段或字段族，并在该阶段保留或使用 |
| `optional` | 当前数据中可能存在，但只在明确请求或配置时进入该阶段 |
| `partial` | 当前有 proxy、聚合或描述性输入，但不足以支持正式研究结论 |
| `unavailable` | 当前链路没有保留或没有输入该字段 |
| `future` | 属于 N5 full、N6 或后续能力，需要单独设计和验证 |

## Cross-Stage Inventory

| Field or family | Full panel | Slim factor table | Evaluation input | Future / optional | Notes |
|---|---|---|---|---|---|
| `stock_code`, `trade_date` | `available` | `available` | `available` | no | Stock-date grain key. Do not print sample stock codes in docs or reports. |
| `market_type` | `available` | `available` | `optional` | optional segment input | Preserved in slim table, but evaluation reads it only if requested as a segment-like input. |
| OHLC price fields: `open`, `high`, `low`, `close`, `prev_close` | `available` | `unavailable` except factor construction inputs before projection | `unavailable` | future/optional if a later diagnostic preserves price fields | Panel-only fields; slim factor validation forbids panel-only price columns. |
| Adjusted execution/price fields: `adj_open`, `adj_close`, `return_1d` | `available` | `unavailable` after slim projection | `available` only through derived forward labels / requested return source | optional if explicitly preserved in a future table | Used to build factors and labels, not retained as raw panel fields in the slim table. |
| Liquidity base fields: `volume`, `amount`, `turnover` | `available` | `amount`, `turnover` are `available`; `volume` is `unavailable` | `optional` only if explicitly requested or transformed into factor/diagnostic input | future for richer liquidity and ADV diagnostics | `amount` and `turnover` can support descriptive proxy work, but do not represent a real capacity model. |
| Market cap fields: `total_market_cap`, `float_market_cap` | `available` | `available` | `optional` only if explicitly requested as risk/neutralization/segment input | future for formal size exposure diagnostics | Available as descriptive context/proxy input, not a Barra exposure or formal risk model. |
| Share count fields: `total_shares`, `a_float_shares`, `share_change_date` | `available` | `unavailable` | `unavailable` | future/optional for capacity/liquidity design | Used to derive turnover in panel construction. |
| Listing and lifecycle fields: `listing_date`, `delisting_date`, `listing_age_days`, `listing_age_trading_days` | `available` | `unavailable` except derived eligibility result | `unavailable` except via `eligible` | future/optional for lifecycle diagnostics | Current evaluation consumes the resulting filters, not full lifecycle fields. |
| Status and trading fields: `is_trading`, `limit_status`, `is_limit_up`, `is_limit_down`, `is_st`, `is_pt`, `eligible_base` | `available` | `unavailable` except derived filters | `unavailable` except via `eligible`, `can_enter`, `can_exit` | future/optional for richer diagnostics | Current tradability inputs are reduced to evaluation filter columns. |
| Evaluation filters: `eligible`, `can_enter`, `can_exit` | `available` | `available` | `available` | no | Used by `TradabilityConfig` and label availability summaries. |
| Price-volume factors: `mom_*`, `rev_*`, `vol_*`, `avg_amount_*`, `turnover_*`, `amihud_*`, `amplitude_*` | calculated from panel, not panel source fields | `available` | `available` when selected as feature columns | extendable by approved factor design | Current default windows are defined in factor code. |
| Forward labels: `forward_return_*` | calculated from panel, not panel source fields | `available` | `available` as labels and return source columns | extendable by approved label design | Labels use future execution-price windows and must remain separate from factor inputs. |
| Industry fields: `csrc_2012_industry_code`, `csrc_2012_industry_name` | `available` | `unavailable` | `unavailable` unless a future evaluation input preserves them | future for sector/industry exposure | Do not describe industry or sector risk as currently available in evaluation artifacts. |
| Benchmark or regime fields | `unavailable` | `unavailable` | `unavailable` | future | Current N5 regime output is an ex-post observation-count proxy, not PIT regime evidence. |

## Full Panel

The full panel is the widest current processed A 股 table. It preserves stock-date keys, daily market fields, adjusted-price inputs, liquidity inputs, market-cap/share-count context, listing-age fields, and derived eligibility/tradability flags.

Current full-panel responsibilities:

- Provide the research-grade stock-date panel for factor and label construction.
- Keep panel-only fields that are useful for audit, quality checks, or future approved diagnostics.
- Avoid acting as the public artifact/display contract.

Current full-panel non-goals:

- It is not a public artifact table.
- It is not an evaluation artifact schema.
- It must not be summarized with row-level private examples in docs.

## Slim Factor Table

The slim factor table is the compact handoff to A 股 evaluation and pipeline flows. It keeps only keys, filters, common controls, factor columns, and forward labels.

Current slim factor base columns:

```text
stock_code
trade_date
market_type
eligible
can_enter
can_exit
amount
turnover
total_market_cap
float_market_cap
```

Current factor and label families:

```text
mom_*
rev_*
vol_*
avg_amount_*
turnover_*
amihud_*
amplitude_*
forward_return_*
```

Panel-only fields such as raw OHLC prices, adjusted prices, listing dates, stock names, and industry names are intentionally not part of the current slim factor table.

## Evaluation Input

The A 股 evaluation adapter reads only the columns needed for a requested run:

- keys: `trade_date`, `stock_code`
- selected feature columns
- selected label columns
- `return_source_column`
- tradability filters: `eligible`, `can_enter`, `can_exit`
- optional `segment_columns`
- optional `risk_column`
- optional neutralization group/exposure columns

This means a field can be present in the slim factor table but not enter a specific evaluation run. For diagnostics-enabled A-share runs, the adapter may preserve only the approved N5 optional context whitelist: `amount`, `turnover`, `total_market_cap`, `float_market_cap`, `avg_amount_*`, `turnover_*`, `amihud_*`, `amplitude_*`, and `market_type`. Industry, sector, stock name, listing-date, raw OHLC, private audit, and full-panel lifecycle fields remain excluded from the current evaluation input.

## N5 Diagnostics Readiness

N5 v1 is partial implementation and contract groundwork. The current diagnostics are post-run descriptive proxy diagnostics; they are not alpha proof, a true risk model, a Barra exposure model, a real capacity model, a backtest, simulation, paper trading, live trading, or investment advice.

| Diagnostic area | Required current input | Current readiness | Limitation | Future input needed |
|---|---|---|---|---|
| Risk path proxy | `portfolio/returns.net_return` | `available` descriptive proxy | Uses realized return path only; not factor exposure or risk model | Formal exposures, covariance/risk model, PIT benchmark/regime context |
| Cost proxy | `portfolio/returns.cost`, `portfolio/returns.gross_portfolio_return`, `portfolio/returns.turnover` | `available` descriptive proxy | Flat bps sensitivity is a scenario proxy only; not tax, slippage, fill, or impact modeling | Approved cost grid, tax/fee layers, slippage/impact assumptions |
| Capacity proxy | `portfolio/returns.turnover`; optional `amount`, `turnover`, `avg_amount_*`, `turnover_*`, `amihud_*`, `amplitude_*` | `partial` descriptive proxy | Uses aggregate optional context only; no row-level private output, ADV participation, or real capacity claim | ADV, liquidity buckets, participation caps, capacity curve design |
| Market cap / liquidity context | `amount`, `turnover`, `total_market_cap`, `float_market_cap`, `market_type` and approved liquidity factor families in slim table | `partial` descriptive proxy in diagnostics-enabled A-share runs | Optional context is preserved only by whitelist and summarized as aggregate proxy; it is not a Barra exposure or formal risk model | Approved full N5 diagnostic design if richer exposure tables are needed |
| Tradability summary | aggregate `eligible`, `can_enter`, `can_exit` derived diagnostics | `partial` descriptive proxy | Current materialized views use aggregate diagnostics, not row-level private observations | Approved table design if full-period tradability diagnostics are needed |
| A-share assumptions | run metadata such as T+1 and no short-sale simulation | `partial` descriptive metadata | Assumption recorded only; no execution simulation | Separate approved simulation/execution design if ever needed |
| Regime proxy | `portfolio/returns` row count | `partial` descriptive proxy | No PIT regime input; ex-post observation count only | PIT regime labels or market-state inputs |
| Industry / sector exposure | industry fields preserved in full panel only | `unavailable` for current evaluation input | Slim factor table and evaluation input do not preserve industry fields by default | Approved preservation of industry/sector fields plus exposure diagnostics design |
| Benchmark-relative diagnostics | none in current evaluation input | `unavailable` | No benchmark return input in current A 股 evaluation run input | PIT benchmark return series and benchmark-relative artifact contract |
| Barra-style or formal risk exposure | none | `future` | Not implemented and not implied by market cap or liquidity fields | Separate formal risk model design, inputs, validation, and artifact contract |
| Real capacity model | none | `future` | Not implemented and not implied by turnover or amount | ADV, order-size assumptions, participation caps, market impact model |

## Maintenance Notes

- If a mechanically checkable artifact table contract changes, update `schema_registry.py` and tests first; this document should explain the resulting availability, not become the sole contract.
- If A 股 pipeline changes which columns enter full panel or slim factor table, update this inventory together with [a-share-csmar-data-manual.md](a-share-csmar-data-manual.md) and `docs/code-map/a-share-data-and-pipeline.md`.
- If N5 full diagnostics are approved later, update this file only after the producer, artifact schema, `manifest.display`, API/download, frontend, tests, and code map are synchronized.
