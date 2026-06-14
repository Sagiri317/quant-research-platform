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

- `contracts/evaluations.py` defines `EvaluationSpec` and
  `EvaluationDiagnostics`.
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

Evaluation reports persist `metrics.parquet`, `grouped_label_values.parquet`,
`ic_summary.parquet`, and `group_spread_summary.parquet` in that order.
Evaluation owns filtering, summary construction, and typed
`evaluation_diagnostics`; compute owns IC, RankIC, grouped label-value, IC
summary, and group-spread kernels; report writers validate and persist frames
without importing compute. Supported evaluation filter modes are `eligible`,
`tradable`, and `unfiltered`; `tradable` is an execution-aware evaluation
diagnostic only, not a backtest, live tradable PnL, or production trading
result.

Prediction bundles record fixed-shape `model_metadata` and
`prediction_metadata` in report manifests. Supported sample filter modes are
`eligible` and `unfiltered`; prediction generation uses feature values only and
does not use labels as prediction inputs. The current prediction path remains a
historical research handoff, not label-free live inference. Missing model or
prediction metadata fails closed in the report writer.

Report manifest payload construction is private to
`reports/report_artifacts.py`. Retired `reports/manifests.py`,
`reports/bundles.py`, report repositories, report stores, and public report
lookup remain outside the active `0.1.0` surface.

## Privacy

Report payloads and errors must not expose local absolute paths, private layer
names beyond approved roots, vendor/source markers, row samples, stock-code
samples, credentials, tokens, or raw tracebacks with local/private context.
