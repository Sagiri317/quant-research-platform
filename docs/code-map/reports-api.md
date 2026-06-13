# Reports And API Code Map

QRP Clean Rewrite `0.1.0` treats evaluation and prediction evidence as private
report outputs behind 0.1 contracts. This file records current implementation
ownership only.

## Flow

```text
research/evaluations
-> reports/report_artifacts.py
-> reports/private_real/<market>/<run_type>/<run_id>
```

## Contract Boundaries

- `contracts/evaluations.py` defines `EvaluationSpec`.
- `contracts/predictions.py` defines `PredictionSpec`.
- `contracts/portfolios.py` defines `PortfolioSpec`.
- `contracts/reports.py` defines `ReportDefinition` and `ReportSpec`.
- `catalog/reports.py` owns `ReportCatalog` and registered report definitions.
- `reports/` owns the report catalog and private report artifact writers.
- Public API/CLI report lookup is not part of the active `0.1.0` surface.
- Future-only report lookup belongs with `report index/show` and
  `ReportRepository` planning.

Persisted stage profile, prediction, and evaluation report artifacts are derived
from `StorageLayout + ReportSpec + run_id`. Report artifact writers do not accept
arbitrary `output_root` or `output_dir` path-selection inputs.

Evaluation reports record `filter_mode` metadata. Supported modes are
`eligible`, `tradable`, and `unfiltered`; `tradable` is an execution-aware
evaluation diagnostic only, not a backtest, live tradable PnL, or production
trading result.

Prediction bundles record `sample_filter_mode` metadata. Supported modes are
`eligible` and `unfiltered`; prediction generation uses feature values only and
does not use labels as prediction inputs. The current prediction path remains a
historical research handoff, not label-free live inference.

## Privacy

Report payloads and errors must not expose local absolute paths, private layer
names beyond approved roots, vendor/source markers, row samples, stock-code
samples, credentials, tokens, or raw tracebacks with local/private context.
