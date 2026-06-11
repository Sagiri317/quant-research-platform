# A-share Data And Workflow Code Map

This map records the current A-share clean-rewrite direction. It does not
authorize private data exposure, website UI work, deployment, workers, schedulers,
external APIs, paper trading, or live trading.

## Private Data Layers

```text
data/raw/<internal-provider>/
-> data/interim/<internal-provider>/
-> markets/implementations/a_share/
-> data/processed/a_share/
-> data/research/a_share/
-> reports/private_real/a_share/
```

Raw files are user-provided local exports. The project does not log in to
vendors, download vendor data, or manage credentials.

## Processed Layout

```text
data/processed/a_share/
  panel/<data_id>/
  annotation/<data_id>/
  factor/<data_id>/
  label/<data_id>/

data/research/a_share/
  <research_dataset_id>/
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
| A-share processed panel/context/universe/masks | `markets/implementations/a_share/processed/` with internal session windows |
| A-share factor formulas | `markets/implementations/a_share/factors/` |
| Forward labels | `markets/implementations/a_share/labels/` |
| Research dataset assembly | `research/datasets` |
| Evaluation | `research/evaluations/` |
| Non-DL prediction bundle | `research/predictions/` and `research/models/` |
| Workflow orchestration | `workflows/` |
| Reports | `reports/` |

## Verification Intent

- Data-layer correctness covers only raw internal-provider CSV input through interim,
  processed A-share panel/annotation/factor/label artifacts, and research
  dataset assembly.
- Correctness evidence uses hand-checkable golden samples, simple reference
  implementations, forced tiny session chunks versus default paths, invariant
  tests, and key-diff diagnostics. Prior materialized runs are comparison
  inputs, not correctness truth.
- Key-diff evidence reports baseline-only, candidate-only, and common-key value
  diagnostics. Expected date deltas are classified against the trading-session
  set, not inferred from candidate artifact dates alone.
- Evaluation, model, prediction, and portfolio outputs are outside the
  data-layer correctness seal and must not be treated as validated by it.
- Private boundary and sanitizer tests remain required before handoff.

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
- Runtime resource policy controls internal processed/research chunk size.
  A-share daily execution chunks by trading-session core windows with read halos:
  next-session halo for panel tradability, past-session warmup for factors, and
  future-session lookahead for labels. Only core rows are written.
- `TimeWindow` is the public logical interval. `SessionWindow` is the internal
  execution interval used by processed materialization and research dataset
  assembly. Public API/CLI surfaces do not expose chunk controls.
