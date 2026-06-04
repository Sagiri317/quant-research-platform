# Contracts And Verification

This is the change-prone contract map for QRP Clean Rewrite `0.1.0`.

## Versions

- Project version: `0.1.0`.
- API/spec/contract version: `0.1`.
- API paths use `/api/0.1/...`.

## Source Map

| Surface | Start here |
|---|---|
| Stable definitions/specs | `src/quant_research_platform/contracts/` |
| Storage layout/manifests/privacy | `src/quant_research_platform/storage/` |
| Catalog registries | `src/quant_research_platform/catalog/` |
| Public data product registry | `src/quant_research_platform/catalog/data_products.py` |
| Internal provider adapters | `src/quant_research_platform/sources/` |
| A-share semantics | `src/quant_research_platform/markets/implementations/a_share/` |
| Research domain | `src/quant_research_platform/research/` |
| Workflows | `src/quant_research_platform/workflows/` |
| Compute backends | `src/quant_research_platform/compute/` |
| Proposal boundaries | `src/quant_research_platform/agents/` |
| API and CLI | `src/quant_research_platform/interfaces/` |
| Reports | `src/quant_research_platform/reports/` |
| Runtime safety | `src/quant_research_platform/runtime/` |

## API Contract

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

## CLI Contract

```text
qrp catalog factors
qrp catalog labels
qrp catalog universes
qrp catalog data-products
qrp data-product interim a_share_daily_bar
qrp market build-processed
qrp research build-dataset
qrp workflow run
qrp report show
```

## Private Roots

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
reports/private_real/
```

Do not expose local absolute paths, private rows, stock-code samples,
vendor markers, credentials, tokens, or raw tracebacks with local/private
context.

## Verification

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests
git diff --check
```

Also run the clean rewrite stale-term gate before commit.
