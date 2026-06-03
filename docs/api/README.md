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
POST /api/0.1/sources/{source_id}/interim
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
| `/sources/{source_id}/interim` | `sources/*` adapters; first active source is `csmar` |
| `/markets/{market_id}/processed-data` | `markets/implementations/*` processed-data services |
| `/research-datasets` | `research/datasets` assembly and validation |
| `/workflows` | `workflows` specs, planner, executor, checkpoints |
| `/reports/{run_id}` | `reports` manifests, summaries, bundles |

## CLI

```text
qrp catalog factors
qrp catalog labels
qrp source build-interim
qrp market build-processed
qrp research build-dataset
qrp workflow run
qrp report show
```

Retired legacy command names are not public clean-rewrite contracts.

## Privacy

API responses, CLI output, manifests, reports, and errors must not expose:

```text
private absolute paths
row samples
stock-code samples
vendor/source markers
credentials/secrets/tokens
raw tracebacks with local/private context
```

Row-level private-derived output belongs only under approved private roots.
