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
- Public data product id is `a_share_daily_bar`.
- Current CLI contract is `qrp catalog`, `qrp data-products`, and
  `qrp research-datasets`.
- `execution_mode` defaults to `execute`; `--dry-run` maps to
  `execution_mode=dry_run`.
- Fixture-backed API/CLI execution is supported; private real-data execution is
  not exposed through API/CLI in `0.1.0`.
- Future-only public-contract work remains outside this slice:
  `processed-data-runs`, `workflow-runs`, `report index/show`,
  `RunRepository`, `ReportRepository`, and `private plugin`.
- Active private local-data CLI is limited to
  `qrp-private schema-inventory --project-root . --output <path>`.
- Report writers derive paths from `StorageLayout + ReportSpec + run_id`.
- Public-safe aggregate evidence is generated from local private aggregate
  verification; full historical `daily_panel` aggregate coverage now spans
  `2014-01-01` through `2025-12-31` without committed row-level output.
- Evaluation and prediction evidence is 2014 smoke scope only; it is not a
  2014-2025 full historical evaluation or prediction run.
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
