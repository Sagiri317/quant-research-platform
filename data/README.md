# QRP Clean Rewrite Data Layout

`data/` is a private boundary by default. It stores private row-level inputs and
approved private research input layers for QRP Clean Rewrite `0.1.0`.

Active layout:

```text
data/
  raw/<internal-provider>/
  interim/<internal-provider>/
  processed/a_share/

reports/
  private_real/<market>/<run_type>/<run_id>/
```

Rules:

- Raw private exports remain in their original form under approved private roots.
- Interim parquet is generated through approved internal adapters and
  catalog/spec writers.
- `data/processed/a_share/` keeps processed market data and approved research
  input layers only.
- `reports/private_real/` is a private boundary by default.
- Private boundary checks do not depend on a suffix convention.
- Prediction, evaluation, diagnostics, factor-generation, and model-output run
  artifacts must be written under `reports/private_real/<market>/<run_type>/<run_id>/`.
- Run reports and model outputs must not be written to `data/processed/`.

Public identifier policy:

- Public surfaces use `data_product_id=a_share_daily_bar`.
- Private row-level evidence, raw file evidence, manifest summaries, report
  summaries, stdout/stderr, and API errors must not expose provider markers,
  local paths, row samples, stock-code samples, credentials, or raw values.
