# A-share Data And Workflow Code Map

This map records the current A-share clean-rewrite direction. It does not
authorize private data exposure, website UI work, deployment, workers, schedulers,
external APIs, paper trading, or live trading.

## Private Data Layers

```text
data_product_id=a_share_daily_bar
-> internal provider adapter
-> data/raw/<internal-provider>/
-> data/interim/<internal-provider>/
-> markets/implementations/a_share/
-> data/processed/a_share/
-> reports/private_real/a_share/
```

Raw files are user-provided local exports. The project does not log in to
vendors, download vendor data, or manage credentials.

## Processed Layout

```text
data/processed/a_share/
  panels/
  contexts/
  universes/
  masks/
    as_of/
    execution/
    evaluation/
  factors/
  labels/
  research_datasets/
```

Canonical processed key:

```text
market_id
asset_id
timestamp
```

`stock_code` and `trade_date` are source-local transient aliases, not the
processed canonical key.

## Current Clean-Rewrite Owners

| Capability | Target owner |
|---|---|
| Raw/interim normalization | `sources/` internal adapters |
| A-share processed panel/context/universe/masks | `markets/implementations/a_share/processed/` |
| A-share factor formulas | `markets/implementations/a_share/factors/` |
| Forward labels | `markets/implementations/a_share/labels/` |
| Research dataset assembly | `research/datasets` |
| Evaluation | `research/evaluations/` |
| Non-DL prediction bundle | `research/predictions/` and `research/models/` |
| Workflow orchestration | `workflows/` |
| Reports | `reports/` |

## Verification Intent

- 2014 processed panel parity-or-explained-delta.
- Prior local 2014-2025 full historical `daily_panel` aggregate coverage;
  full evidence/report regeneration remains pending after the factor catalog
  refactor.
- 2014-2025 factor value parity-or-explained-delta.
- 2014-2025 label value parity-or-explained-delta.
- 2014-2025 masks/universes parity.
- 2014-2025 research dataset row/key/column parity.
- 2014 evaluation and prediction smoke parity only.
- Private boundary and sanitizer tests.

## Current Local Coverage Status

- 2014 processed-layer parity-or-explained-delta completed.
- 2014 ResearchDataset, evaluation, and prediction smoke parity completed.
- 2014-2025 factor/label-supporting processed layers were verified through
  spec-driven writers.
- 2014-2025 `a_share_price_volume_forward_returns` ResearchDataset materialized
  through the ResearchDataset assembler.
- Full historical `daily_panel` aggregate coverage is verified for
  `2014-01-01` through `2025-12-31` by local private aggregate verification
  without committing row-level output artifacts.
- Evaluation and prediction remain 2014 smoke evidence; they have not been
  expanded into full historical 2014-2025 runs.
- Evaluation supports `eligible`, `tradable`, and `unfiltered` filter modes.
  `tradable` means `eligible and can_enter and can_exit` and is an
  execution-aware evaluation diagnostic only, not a backtest, live tradable PnL,
  or production trading result.
- Model training and historical prediction handoff support only `eligible` and
  `unfiltered` sample filter modes. They do not use `can_enter` or `can_exit`
  for sample filtering.

## Narrow Follow-up Notes

- `stock_master` panel as-of joins use processed keys
  `asset_id + stock_effective_date`. Exact duplicate rows are deduped
  deterministically and conflicting duplicate keys fail closed.
- `market_cap` panel as-of joins use processed keys `asset_id + timestamp`.
  Exact duplicate rows are deduped deterministically and conflicting duplicate
  keys fail closed.
- `eligible[t]` is the single EOD-as-of-t ResearchDataset universe mask. It is
  known after signal-date close and intended for t+1 open research
  prediction/evaluation. It may combine pre-open/reference information and
  t-day EOD trading/liquidity information.
- ST/PT timing exactness remains future P2. The current mask does not split
  pre-open reference status, trade-date effective status, EOD liquidity, and
  next-open-known ST/PT status.
- `can_enter[t]` is false only when the next trading row is one-word limit up
  or unavailable/non-trading; `can_exit[t]` is false only when the next trading
  row is one-word limit down or unavailable/non-trading. Non-one-word limit
  up/down rows remain tradable at the open-price abstraction used in this
  research batch.
