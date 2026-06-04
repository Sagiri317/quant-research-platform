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
-> reports/private_real/<market>/<run_type>/<run_id>
```

## Contract Boundaries

- `contracts/evaluations.py` defines `EvaluationSpec`.
- `contracts/predictions.py` defines `PredictionSpec`.
- `contracts/portfolios.py` defines `PortfolioSpec`.
- `contracts/reports.py` defines `ReportDefinition` and `ReportSpec`.
- `catalog/reports.py` owns `ReportCatalog` and registered report definitions.
- `reports/` owns report manifests, summaries, and bundles.
- Public API/CLI report lookup is not part of the active `0.1.0` surface.
- Future-only report lookup belongs with `report index/show` and
  `ReportRepository` planning.

Persisted stage profile, prediction, and evaluation report outputs are derived
from `StorageLayout + ReportSpec + run_id`. Report writers do not accept
arbitrary `output_root` or `output_dir` path-selection inputs.

## Privacy

Report payloads and errors must not expose local absolute paths, private layer
names beyond approved roots, vendor/source markers, row samples, stock-code
samples, credentials, tokens, or raw tracebacks with local/private context.
