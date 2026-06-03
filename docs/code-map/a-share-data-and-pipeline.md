# A-share Data And Workflow Code Map

This map records the current A-share clean-rewrite direction. It does not
authorize private data exposure, website UI work, deployment, workers, schedulers,
external APIs, paper trading, or live trading.

## Private Data Layers

```text
data/raw/csmar/
-> sources/csmar/
-> data/interim/csmar/
-> markets/implementations/a_share/
-> data/processed/a_share/
-> reports/private_real/a_share/
```

Raw CSMAR files are user-provided local exports. The project does not log in to
CSMAR, download vendor data, or manage credentials.

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

`stock_code` and `trade_date` are source/migration fields, not the processed
canonical key.

## Current Migration Targets

| Capability | Target owner |
|---|---|
| CSMAR raw/interim normalization | `sources/csmar/` |
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
- 2014 factor value parity-or-explained-delta.
- 2014 label value parity-or-explained-delta.
- Masks/universes parity.
- Research dataset row/key/column parity.
- Evaluation and prediction smoke parity.
- Private boundary and sanitizer tests.

## Current Local Migration Status

- 2014 processed-layer parity-or-explained-delta completed.
- 2014 ResearchDataset, evaluation, and prediction smoke parity completed.
- 2014-2025 factor/label-supporting processed layers migrated through
  spec-driven writers.
- 2014-2025 `a_share_price_volume_forward_returns` ResearchDataset materialized
  through the ResearchDataset assembler.
- Full historical `daily_panel` remains generated/validated for 2014 only in
  this clean-rewrite slice.
