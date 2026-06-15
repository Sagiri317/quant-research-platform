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
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

Route ownership:

| Route | Owner |
|---|---|
| `/catalog/factors` | `catalog.factors.FactorCatalog` and `contracts.factors.FactorDefinition` |
| `/catalog/labels` | `catalog.labels.LabelCatalog` and `contracts.labels.LabelDefinition` |
| `/research-datasets/{dataset_id}/requirements` | `interfaces.services.describe_research_dataset_requirements` read-only requirement query |

The research dataset requirements route is read-only. It reports the approved
dataset id and required input data ids; it does not build or write dataset
artifacts. Source/interim/processed materialization is internal and is not
exposed through API/CLI in `0.1.0`.

## CLI

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

The CLI does not expose private real-data flags, source roots, input roots, data
roots, provider paths, materialization commands, or overwrite/force controls.

Retired legacy command names are not public clean-rewrite contracts.

## Naming Convention

- API resource segment: kebab-case + plural, for example `/catalog` and
  `/research-datasets`.
- API path parameter: snake_case, for example `{dataset_id}`.
- ID value: snake_case, for example `a_share` and
  `a_share_price_volume_forward_returns`.
- JSON field: snake_case, for example `dataset_id` and `input_data_ids`.
- CLI group / command / option: kebab-case, for example `catalog`,
  `research-datasets`, and `requirements`.
- Python module/function: snake_case, for example
  `describe_research_dataset_requirements`.
- Do not convert business ID values such as
  `a_share_price_volume_forward_returns` to kebab-case.

## Future-Only Items

The following are not active `0.1.0` public contracts: `processed-data-runs`,
`workflow-runs`, `report index/show`, `RunRepository`, `ReportRepository`, and
`private plugin`.

## Private Local CLI

The approved private local CLI commands are:

```text
qrp-private schema-inventory --project-root . --output <path>
qrp-private evidence run
```

`qrp-private schema-inventory` writes sanitized aggregate schema metadata for
sealed dataset roots. It does not scan `data/raw` or `reports/private_real`, and
it does not run private import or private rebuild.

`qrp-private evidence run` is a private local verification harness for local
smoke/evidence generation. It is not a public CLI, not a public execution mode,
not a public evaluate, predict, or report command, not paper or live trading,
does not promise returns, and does not output row-level private data.

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
