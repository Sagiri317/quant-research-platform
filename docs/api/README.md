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
POST /api/0.1/data-products/{data_product_id}/interim-runs
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

Route ownership:

| Route | Owner |
|---|---|
| `/catalog/factors` | `catalog.factors.FactorCatalog` and `contracts.factors.FactorDefinition` |
| `/catalog/labels` | `catalog.labels.LabelCatalog` and `contracts.labels.LabelDefinition` |
| `/catalog/universes` | `contracts.universes.UniverseDefinition` metadata |
| `/catalog/data-products` | `catalog.data_products.DataProductRegistry` public records |
| `/data-products/{data_product_id}/interim-runs` | `interfaces.services.run_data_product_interim` fixture-backed registered execution or dry-run validation for approved data product specs |
| `/research-datasets/{dataset_id}/requirements` | `interfaces.services.describe_research_dataset_requirements` read-only requirement query |

The data-product POST route accepts `DataProductInterimRunRequest` with
`execution_mode` and optional `run_id`. `execution_mode` defaults to `execute`;
`dry_run` validates catalog/spec/writer boundaries and returns a sanitized
validation summary only. `execute` runs the minimal fixture-backed registered
path. Private real-data execution is not exposed through API/CLI in `0.1.0`.

The research dataset requirements route is read-only. It reports the approved
dataset id and required processed data ids; it does not build or write dataset
artifacts.

## CLI

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

CLI `qrp data-products run-interim ...` defaults to `execution_mode=execute`;
pass `--dry-run` to request `execution_mode=dry_run`. `--run-id` is an advanced
public-safe logical run id and is validated through the shared run id contract.
The CLI does not expose `--execute`, private real-data flags, source roots,
input roots, data roots, or provider paths.

Retired legacy command names are not public clean-rewrite contracts.

## Naming Convention

- API resource segment: kebab-case + plural, for example `/data-products`,
  `/interim-runs`, and `/research-datasets`.
- API path parameter: snake_case, for example `{data_product_id}`,
  `{dataset_id}`, and `{run_id}`.
- ID value: snake_case, for example `a_share_daily_bar`, `a_share`, and
  `a_share_price_volume_forward_returns`.
- JSON field: snake_case, for example `data_product_id`, `execution_mode`, and
  `run_id`.
- CLI group / command / option: kebab-case, for example `data-products`,
  `run-interim`, `research-datasets`, `--dry-run`, and `--run-id`.
- Python module/function: snake_case, for example `run_data_product_interim`
  and `describe_research_dataset_requirements`.
- Do not convert business ID values such as `a_share_daily_bar` to kebab-case.

## Future-Only Items

The following are not active `0.1.0` public contracts: `processed-data-runs`,
`workflow-runs`, `report index/show`, `RunRepository`, `ReportRepository`,
`private plugin`, and `qrp-private`.

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
