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
POST /api/0.1/data-products/{data_product_id}/interim-runs
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

The API is local-only. Catalog routes return registered metadata. The data
product POST route accepts `execution_mode`, defaults to `execute`, and runs the
minimal fixture-backed registered path; `dry_run` performs sanitized
catalog/spec/writer validation without row-level writes. The research dataset
requirements route is read-only and does not build artifacts. The API does not
expose private real-data execution, deployment, scheduler, worker, paper/live
trading, external data, broker/exchange, or website UI contracts.

## CLI Flow

```text
qrp catalog factors
qrp catalog labels
qrp catalog universes
qrp catalog data-products
qrp data-products run-interim <data_product_id>
qrp data-products run-interim <data_product_id> --dry-run
qrp data-products run-interim <data_product_id> --run-id <run_id>
qrp research-datasets requirements <dataset_id>
```

`qrp data-products run-interim ...` defaults to `execution_mode=execute`;
`--dry-run` maps to `execution_mode=dry_run`, and `--run-id` is validated by the
shared public-safe run id contract. No private-source root flags are part of the
`0.1.0` CLI contract.

Retired commands and endpoints are not current clean-rewrite contracts.

Future-only items remain outside the active surface: `processed-data-runs`,
`workflow-runs`, `report index/show`, `RunRepository`, `ReportRepository`,
and `private plugin`.

Private local schema inventory is exposed separately through
`qrp-private schema-inventory --project-root . --output <path>`. It scans only
`data/interim` and `data/processed` and emits private aggregate schema
inventory metadata. This inventory may include logical source paths or source
field names, so it is not the same thing as public-safe aggregate evidence.

## Narrow Follow-up Notes

- Workflow summaries currently expose `completed_nodes`; a future summary model
  may separate `planned_nodes` from `completed_nodes` so dry-run plans are not
  represented as completed work.
- Storage writer root validation currently validates spec-shaped legal suffixes;
  a future implementation should make this project-root-relative, likely by
  passing `StorageLayout.project_root` or equivalent through writer validation
  paths.
