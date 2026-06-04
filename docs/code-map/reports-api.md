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
- `contracts/reports.py` defines `ReportDefinition` and `ReportSpec`.
- `catalog/reports.py` owns `ReportCatalog` and registered report definitions.
- `reports/` owns report manifests, summaries, and bundles.
- `interfaces/api` exposes local report lookup through `/api/0.1/reports/{run_id}`.
- `interfaces/cli` exposes local report lookup through `qrp report show`.

Persisted stage profile, prediction, and evaluation report outputs are derived
from `StorageLayout + ReportSpec + run_id`. Optional `output_root` arguments in
tests are accepted only when they equal the derived path; they are not business
or API/CLI path-selection inputs.

## Privacy

Report payloads and errors must not expose local absolute paths, private layer
names beyond approved roots, vendor/source markers, row samples, stock-code
samples, credentials, tokens, or raw tracebacks with local/private context.
