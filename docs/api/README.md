# QRP API 0.1

This document records the clean rewrite API and CLI contract. It is local-only
and does not describe deployment, external access, broker/exchange connectivity,
paper trading, live trading, or investment advice.

## Versioning

- Project version: `0.1.0`.
- API/spec/contract version: `0.1`.
- Paths use `/api/0.1/...`; do not add a version-letter prefix.

## Local API Routes

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/catalog/universes
GET  /api/0.1/catalog/data-products
POST /api/0.1/data-products/a_share_daily_bar/interim
POST /api/0.1/markets/{market_id}/processed-data
POST /api/0.1/research-datasets
POST /api/0.1/workflows
GET  /api/0.1/reports/{run_id}
```

Route ownership:

| Route | Owner |
|---|---|
| `/catalog/factors` | `catalog.factors.FactorCatalog` and `contracts.factors.FactorDefinition` |
| `/catalog/labels` | `catalog.labels.LabelCatalog` and `contracts.labels.LabelDefinition` |
| `/catalog/universes` | `contracts.universes.UniverseDefinition` metadata |
| `/catalog/data-products` | `catalog.data_products.DataProductRegistry` public records |
| `/data-products/a_share_daily_bar/interim` | `interfaces.services` fixture-backed registered workflow execution or dry-run validation for approved data product specs |
| `/markets/{market_id}/processed-data` | `interfaces.services` dry-run validation for approved processed specs |
| `/research-datasets` | `interfaces.services` dry-run validation for ResearchDataset specs |
| `/workflows` | `workflows` specs, planner, executor, checkpoints through `interfaces.services` |
| `/reports/{run_id}` | structured report metadata response; no local folder scan in the adapter |

The data-product POST route accepts `DataProductInterimRequest` with
`execution_mode` and optional `run_id`. `execution_mode` defaults to `execute`;
`dry_run` validates catalog/spec/workflow/writer boundaries and returns a
sanitized validation summary only. `execute` runs the minimal fixture-backed
registered workflow path. Private real-data execution is not exposed through
API/CLI in `0.1.0`.

Other POST routes are thin service adapters in `0.1.0`: they validate approved
spec/catalog boundaries and return sanitized summaries. They do not claim full
remote execution, deployment, scheduler, worker, or public data access.

## CLI

```text
qrp catalog factors
qrp catalog labels
qrp catalog universes
qrp catalog data-products
qrp data-product interim a_share_daily_bar
qrp data-product interim a_share_daily_bar
qrp market build-processed
qrp research build-dataset
qrp workflow run
qrp report show
```

CLI `qrp data-product ...` defaults to `execution_mode=execute`; pass
`--dry-run` to request `execution_mode=dry_run`. The CLI does not expose
`--execute`, private real-data flags, source roots, input roots, data roots, or
provider paths.

Retired legacy command names are not public clean-rewrite contracts.

## Privacy

API responses, CLI output, manifests, reports, and errors must not expose:

```text
private absolute paths
row samples
stock-code samples
vendor markers
credentials/secrets/tokens
raw tracebacks with local/private context
```

Row-level private-derived output belongs only under approved private roots.
