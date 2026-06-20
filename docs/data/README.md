# Data Layout

This file records public-safe data layout rules for QRP Clean Rewrite `0.1.0`.
It does not contain vendor field samples or private row-level examples.

## Private Roots

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
data/research/a_share/
reports/private_research/
```

`data/` and `reports/private_research/` are private by default. Privacy is enforced
by approved roots and sanitizer policy, not by a suffix naming convention.

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
reports/private_research/<market>/<run_type>/<run_id>/
```

They must not be written to `data/processed/`.
Writers derive these roots from `StorageLayout + ReportSpec + run_id`; API/CLI
and business services do not accept arbitrary public output roots.

## Public Identifier Policy

Public-safe evidence, manifest summaries, report summaries, stdout/stderr, and
API errors use semantic identifiers such as `data_id`, `artifact_id`,
`workflow_id`, or `research_dataset_id`; they must not expose provider markers,
local paths, row samples, stock-code samples, credentials, or raw values.
