# Data Layout

This file records public-safe data layout rules for QRP Clean Rewrite `0.1.0`.
It does not contain vendor field samples or private row-level examples.

## Private Roots

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
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

Each processed-data object and research dataset is addressed by its artifact
root. Business, research, market, workflow, and source layers must use
`storage.datasets` APIs for dataset IO instead of manually addressing dataset
artifact internals.

## Canonical Columns

```text
market_id
asset_id
timestamp
session_date
available_at
```

Provider fields such as `stock_code` and `trade_date` may be retained only as
source-local transient aliases before persisted interim output.

## Run Outputs

Prediction, evaluation, diagnostics, factor-generation, and model-output run
artifacts belong under:

```text
reports/private_real/<market>/<run_type>/<run_id>/
```

They must not be written to `data/processed/`.
Writers derive these roots from `StorageLayout + ReportSpec + run_id`; API/CLI
and business services do not accept arbitrary public output roots.

## Public Identifier Policy

Public surfaces use `data_product_id=a_share_daily_bar`. Public-safe evidence,
manifest summaries, report summaries, stdout/stderr, and API errors must not
expose provider markers, local paths, row samples, stock-code samples,
credentials, or raw values.
