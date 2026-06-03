# Data Layout

This file records public-safe data layout rules for QRP Clean Rewrite `0.1.0`.
It does not contain vendor field samples or private row-level examples.

## Private Roots

```text
data/raw/csmar/
data/interim/csmar/
data/processed/a_share/
reports/private_real/
```

`data/` and `reports/private_real/` are private by default. Privacy is enforced
by approved roots and sanitizer policy, not by a suffix naming convention.

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

Each processed-data object uses:

```text
data/processed/<market>/<category>/<processed_data_id>/
  data/
  _manifest/
    manifest.json
```

Each research dataset uses:

```text
data/processed/<market>/research_datasets/<dataset_id>/
  spec.json
  data/
  _manifest/
    manifest.json
```

## Canonical Columns

```text
market_id
asset_id
timestamp
session_date
available_at
```

Source fields such as `stock_code` and `trade_date` may be retained only as
source/migration fields.

## Run Outputs

Prediction, evaluation, diagnostics, factor-generation, and model-output run
artifacts belong under:

```text
reports/private_real/<market>/<run_type>/<run_id>/
```

They must not be written to `data/processed/`.

## Source Identifier Policy

`csmar` is a public source identifier for catalog metadata and API routes.
Public-safe evidence, manifest summaries, report summaries, stdout/stderr, and
API errors must not expose source-row markers, local paths, row samples,
stock-code samples, credentials, or raw values.
