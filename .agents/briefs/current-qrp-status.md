# Current QRP Status

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
- Current CLI contract is `qrp catalog/data-product/market/research/workflow/report`.
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

- Continue porting the A-share data product, processed panel, factor/label,
  research dataset, evaluation, prediction, report, and workflow slices into the
  0.1 topology with test-first changes.

## Hard Reminders

- No frontend, deployment, scheduler, worker, model registry, feature store,
  broker/exchange connector, external data API, paper trading, or live trading
  in this clean rewrite slice.
- Do not delete `AGENTS.md` or `.agents/`.
- Run narrow tests first; broader checks come after the touched surface is
  internally consistent.
