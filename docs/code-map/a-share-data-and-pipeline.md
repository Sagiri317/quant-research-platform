# A-share Data And Workflow Code Map

This map records the current A-share clean-rewrite direction. It does not
authorize private data exposure, website UI work, deployment, workers, schedulers,
external APIs, simulated order execution, or real-money order execution.

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
| A-share daily facts | `markets/implementations/a_share/processed/panels.py` |
| A-share context/universe/masks/execution annotations | `markets/implementations/a_share/processed/annotations.py` |
| A-share factor formulas | `markets/implementations/a_share/factors/` |
| Forward labels | `markets/implementations/a_share/labels/` |
| Research dataset assembly | `research/datasets` |
| Evaluation | `research/evaluations/` |
| Non-DL prediction bundle | `research/predictions/` and `research/models/` |
| Workflow orchestration | `workflows/` |
| Reports | `reports/` |

The research profile composition root is `catalog/profiles.py`.
The public research profile composition root returns the A-share internal-provider profile.
A-share owner modules expose field, processed, factor, label, and research
dataset catalogs; source/interim catalogs are composed only in the research
profile.

## Verification Intent

- Data-layer correctness covers only raw internal-provider CSV input through interim,
  processed A-share panel/annotation/factor/label artifacts, and research
  dataset assembly.
- First-stage data-layer closure is structural and contract-level: source/raw
  to interim, A-share fact panel, A-share annotations/masks/context, factor
  data, label data, and research dataset assembly are closed under explicit
  catalog/profile ownership.
- `daily_panel` is a fact layer. It excludes derived `eligible*`,
  `next_day*`, `can_enter`, and `can_exit` masks.
- A-share trading semantics are owned by processed annotations. Label,
  evaluation, model, and prediction paths consume annotation outputs instead of
  recomputing ST, suspension, limit, eligibility, or execution rules.
- The research dataset assembler is generic and market-agnostic. It requires
  `base_input_data_id`, reads the base first, left-joins remaining processed
  inputs on `market_id, asset_id, timestamp`, and emits columns in primary-key,
  feature, label, filter, context order. `key_coverage` is normalized and
  persisted for every declared research input in the dataset spec payload and
  manifest; the base input remains fixed to `strict`.
- Correctness evidence uses hand-checkable golden samples, simple reference
  implementations, forced tiny session chunks versus default paths, invariant
  tests, and key-diff diagnostics. Prior materialized runs are comparison
  inputs, not correctness truth.
- `price_volume` rolling moments use the QRP window-local contract. `std` is
  rolling sample standard deviation with `ddof=1`. `skew` and `kurt` use the
  adjusted sample skewness and adjusted excess kurtosis formulas. Their
  near-zero policy uses `moment_2 = mean((x - mean(x))^2)`, the population
  second central moment, not sample variance with `ddof=1`; for skew/kurt
  only, `moment_2 <= _ROLLING_MOMENT_NEAR_ZERO_VARIANCE = 1e-12` emits `NaN`.
  pandas rolling is a well-conditioned cross-check only; the test-only QRP
  reference is the oracle for zero-moment_2, near-zero-moment_2, NaN, warmup,
  and session-chunk boundary behavior under the shared tolerance.
- Current price-volume factor windows and forward-return label horizons use
  per-asset valid-bar semantics. In column names such as `mom_20d`,
  `turnover_5d`, and `forward_return_10d`, `d` means valid daily bars from
  each asset's own available price-volume record sequence, not calendar days or
  market-wide trading sessions. Missing/suspended rows are not forward-filled,
  backfilled, or filled with zero, and chunk boundaries must not change values
  or `NaN` positions. Future calendar/session semantics require distinct names
  such as `calendar_mom_20d` or `session_forward_return_10d`.
- Key-diff evidence reports baseline-only, candidate-only, and common-key value
  diagnostics. Expected date deltas are classified against the trading-session
  set, not inferred from candidate artifact dates alone.
- Evaluation, model, prediction, and portfolio outputs are outside the
  data-layer correctness seal and must not be treated as validated by it.
- Evaluation/model/prediction diagnostics and registries are outside the
  first-stage data-layer closure. Their demo catalogs are research profile
  components only.
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
- The final `price_volume` rolling-moment follow-up reruns only the bounded
  2014-2016 raw-start smoke unless the corresponding handoff notes say
  otherwise; fresh 2014-2025 full-horizon rerun remains an
  overnight/milestone validation before merge/main/demo.
- Full-horizon private rerun is not a completion requirement for the
  first-stage data-layer closure unless a generated run artifact explicitly
  states it was rerun.
- Evaluation supports `eligible`, `tradable`, and `unfiltered` filter modes.
  `tradable` means `eligible and can_enter and can_exit` and is an
  execution-aware evaluation diagnostic only, not a backtest, real-money-executable PnL,
  or production order execution result.
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
- `eligible[t]` is emitted by A-share processed annotations as the single
  EOD-as-of-t ResearchDataset universe mask. It is known after signal-date
  close and intended for t+1 open research prediction/evaluation. It may
  combine pre-open/reference information and t-day EOD trading/liquidity
  information.
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
  next-session halo for daily panel construction and annotation execution masks,
  per-asset valid daily bar warmup for factors, and per-asset valid daily bar
  future lookahead for labels. Only core rows are written.
- `TimeWindow` is the public logical interval. `SessionWindow` is the internal
  execution interval used by processed materialization and research dataset
  assembly. Public API/CLI surfaces do not expose chunk controls.
