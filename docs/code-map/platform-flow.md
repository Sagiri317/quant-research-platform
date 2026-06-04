# Platform Flow Code Map

This map records the QRP Clean Rewrite `0.1.0` implementation flow. It is a
current code-reading map, not a roadmap or historical closeout.

## Flow

```text
catalog/data_products -> data_product_id=a_share_daily_bar
-> internal provider adapter
-> private raw/interim data roots
-> markets/implementations/a_share
-> data/processed/a_share
-> ResearchDatasetSpec -> research/datasets
-> research/evaluations
-> reports/private_real
-> WorkflowSpec -> workflows
-> interfaces/api and interfaces/cli
```

## Package Ownership

| Package | Responsibility |
|---|---|
| `contracts/` | Stable 0.1 definitions and specs |
| `storage/` | Layout, parquet, manifests, partitions, checksums, privacy |
| `catalog/` | Queryable source of truth for platform objects |
| `sources/` | Internal provider adapters |
| `markets/` | Market-specific semantics, first active market: `a_share` |
| `research/` | Factors, labels, features, datasets, evaluations, models, predictions, portfolios |
| `workflows/` | Simple DAG specs, planner, executor, checkpoints |
| `compute/` | DataFrame and tensor compute backends |
| `agents/` | Proposal-only factory/planner boundaries |
| `interfaces/` | Local CLI and API adapters |
| `reports/` | Report manifests, summaries, and bundles |
| `runtime/` | Privacy, sanitization, logging, profiling, errors |

## API Flow

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

The API is local-only. Catalog routes return registered metadata. POST routes
call the shared thin service layer for sanitized dry-run spec validation; they
are not deployment, scheduler, worker, or remote execution contracts. The API
does not expose paper/live trading, external data, broker/exchange, or website
UI contracts.

## CLI Flow

```text
qrp catalog factors
qrp catalog labels
qrp catalog universes
qrp catalog data-products
qrp data-product build-interim --data-product a_share_daily_bar
qrp market build-processed
qrp research build-dataset
qrp workflow run
qrp report show
```

Retired commands and endpoints are not current clean-rewrite contracts.
