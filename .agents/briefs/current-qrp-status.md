# Current QRP Status

P0/P1/P2/P3 architecture and rebuild follow-ups are tracked in
`docs/architecture/open-issues-register.md`.

This is a short current task context primer. It is not a Git log, changelog, or
replacement for `PROJECT_STATUS.md`.

## Current Phase

- QRP Clean Rewrite `0.1.0`.
- Current architecture style: Stable Core + High-Velocity Research Edge.
- Current work stays local-only and centered on contracts, storage layout,
  catalogs, A-share market semantics, research workflows, reports, and
  private-boundary reliability.

## Current Implementation Context

- Target source topology is `contracts`, `storage`, `catalog`, `sources`,
  `markets`, `research`, `workflows`, `compute`, `agents`, `interfaces`,
  `reports`, and `runtime`.
- Project version is `0.1.0`; API/spec/contract version is `0.1`.
- API paths use `/api/0.1/...`; do not add a version-letter prefix.
- Current public CLI contract is `qrp catalog factors`,
  `qrp catalog labels`, and `qrp research-datasets requirements`.
- Source/interim/processed materialization is internal; private real-data
  execution is not exposed through API/CLI in `0.1.0`.
- Future-only public-contract work remains outside this slice:
  `processed-data-runs`, `workflow-runs`, `report index/show`,
  `RunRepository`, `ReportRepository`, and `private plugin`.
- Active private local-data CLI is limited to
  `qrp-private schema-inventory --project-root . --output <path>`.
- Report writers derive paths from `StorageLayout + ReportSpec + run_id`.
- Runtime resource planning is split from market-session planning:
  `ResourceChunkPlanner` derives internal resource chunk sizes, while
  `SessionWindowPlanner` derives trading-session core/read windows and halos.
- Processed A-share materialization and research dataset assembly are
  session-chunked internally; public API/CLI surfaces do not expose chunk-size,
  year, partition, force, overwrite, or dependency-policy controls.
- First-stage data-layer closure is the active sealed path:
  `source/raw -> interim -> A-share processed facts -> A-share annotations/masks/context -> factor data -> label data -> research dataset`.
- `catalog/profiles.py` is the local composition root. The public local profile
  builder is `local_a_share_csmar_catalogs()` returning `CatalogBundle`;
  retired `catalog/defaults.py`, `default_catalogs()`, and `PlatformCatalogs`
  are replaced and are no longer active public entrypoints.
- A-share owner modules are split by responsibility. Stable IDs remain in
  `markets/implementations/a_share/ids.py`; calendar helpers remain in
  `markets/implementations/a_share/calendar.py`.
- `daily_panel` is the A-share fact layer. It does not emit derived
  `eligible*`, `next_day*`, `can_enter`, or `can_exit` masks.
- A-share trading semantics are owned by
  `markets/implementations/a_share/processed/annotations.py`; labels,
  evaluation, model, and prediction code must consume annotation outputs rather
  than recomputing ST, suspension, limit, eligibility, or execution rules.
- `ResearchDatasetSpec.base_input_data_id` is required. The assembler is
  market-agnostic, reads the explicit base first, left-joins the other inputs
  on `market_id, asset_id, timestamp`, and orders output columns as primary
  key, features, labels, filters, context.
- Data-layer correctness evidence is evaluated from current semantics:
  golden/reference tests, session-chunk parity, invariants, and key-diff
  diagnostics. Older local materialized runs are comparison inputs only, not
  correctness baselines.
- Prior local private aggregate verification covered full historical
  `daily_panel` aggregate coverage for `2014-01-01` through `2025-12-31`
  without committed row-level output; full evidence/report regeneration remains
  pending after the factor catalog refactor.
- Evaluation and prediction evidence is 2014 smoke scope only; it is not a
  2014-2025 full historical evaluation or prediction run.
- Evaluation/model/prediction registries, diagnostics, workflow execute/resume,
  dashboards, high-frequency, multi-market implementation, and paper/live
  trading remain outside the first-stage data-layer closure. Demo
  evaluation/model/prediction catalogs are local profile components only.
- Full-horizon private rerun is not required for the first-stage structural
  closure unless a generated run artifact explicitly records that it occurred.
- `eligible[t]` is the single EOD-as-of-t ResearchDataset universe mask, known
  after signal-date close and intended for t+1 open research prediction and
  evaluation. Model and prediction default sample filtering may use
  `eligible`, but they do not use `can_enter` or `can_exit`.
- Before the next PE/full-data run, processed panel as-of joins should fail
  closed on conflicting duplicate `stock_master` and `market_cap` keys while
  exact duplicate rows are deterministically deduped.
- A-share is the first enabled market. Other markets are future design only.
- `agents/` is proposal-only in this phase; it does not call external LLM APIs
  and does not write private row-level data.

## Private Boundaries

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
reports/private_real/
```

Do not expose raw/interim/processed private rows, private reports, vendor
markers, local absolute paths, credentials, or stock-code samples.

## Current Next Step

- Run final pre-merge verification gates, review the diff, and commit/push only
  if no required blocker remains.

## Hard Reminders

- No frontend, deployment, scheduler, worker, model registry, feature store,
  broker/exchange connector, external data API, paper trading, or live trading
  in this clean rewrite slice.
- Do not delete `AGENTS.md` or `.agents/`.
- Run narrow tests first; broader checks come after the touched surface is
  internally consistent.
