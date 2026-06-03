# Reports And API Code Map

QRP Clean Rewrite `0.1.0` treats evaluation and prediction evidence as private
report outputs behind 0.1 contracts. This file records current implementation
ownership only.

## Flow

```text
research/evaluations
-> reports/manifests.py
-> reports/summaries.py
-> reports/bundles.py
-> interfaces/api /api/0.1/reports/{run_id}
-> interfaces/cli qrp report show
```

## Contract Boundaries

- `contracts/evaluations.py` defines `EvaluationSpec`.
- `contracts/predictions.py` defines `PredictionSpec`.
- `contracts/portfolios.py` defines `PortfolioSpec`.
- `reports/` owns report manifests, summaries, and bundles.
- `interfaces/api` exposes local report lookup through `/api/0.1/reports/{run_id}`.
- `interfaces/cli` exposes local report lookup through `qrp report show`.

## Privacy

Report payloads and errors must not expose local absolute paths, private layer
names beyond approved roots, vendor/source markers, row samples, stock-code
samples, credentials, tokens, or raw tracebacks with local/private context.
