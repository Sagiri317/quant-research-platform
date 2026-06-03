# Platform Flow Code Map

This map records the QRP Clean Rewrite `0.1.0` implementation flow. It is a
current code-reading map, not a roadmap or historical closeout.

## Source Flow

```text
sources/csmar
-> data/raw/csmar
-> data/interim/csmar
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
| `sources/` | Source adapters, first active adapter: `csmar` |
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
POST /api/0.1/sources/{source_id}/interim
POST /api/0.1/markets/{market_id}/processed-data
POST /api/0.1/research-datasets
POST /api/0.1/workflows
GET  /api/0.1/reports/{run_id}
```

The API is local-only. It does not expose deployment, paper/live trading,
external data, broker/exchange, scheduler, worker, or website UI contracts.

## CLI Flow

```text
qrp catalog factors
qrp catalog labels
qrp source build-interim
qrp market build-processed
qrp research build-dataset
qrp workflow run
qrp report show
```

Retired commands and endpoints are not current clean-rewrite contracts.
