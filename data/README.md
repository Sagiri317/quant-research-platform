# QRP Clean Rewrite Data Layout

`data/` is a private boundary by default. It stores private row-level inputs and
approved private research input layers for QRP Clean Rewrite `0.1.0`.

Active layout:

```text
data/
  raw/csmar/
  interim/csmar/
  processed/a_share/

reports/
  private_real/<market>/<run_type>/<run_id>/
```

Rules:

- `data/raw/csmar/` keeps source exports in their original form.
- `data/interim/csmar/` keeps canonical interim parquet generated through the
  approved source adapter and catalog/spec writer.
- `data/processed/a_share/` keeps processed market data and approved research
  input layers only.
- `reports/private_real/` is a private boundary by default.
- Private boundary checks do not depend on a suffix convention.
- Prediction, evaluation, diagnostics, factor-generation, and model-output run
  artifacts must be written under `reports/private_real/<market>/<run_type>/<run_id>/`.
- Run reports and model outputs must not be written to `data/processed/`.

Source identifier policy:

- `csmar` is a public source identifier for catalog metadata and API routes.
- Private row-level evidence, raw file evidence, manifest summaries, report
  summaries, stdout/stderr, and API errors must not expose source-row markers,
  local paths, row samples, stock-code samples, credentials, or raw values.
