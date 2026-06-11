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
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

## CLI Contract

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

Future-only public contract items: `processed-data-runs`, `workflow-runs`,
`report index/show`, `RunRepository`, `ReportRepository`, and `private plugin`.

Private local schema inventory is exposed only as:

```text
qrp-private schema-inventory --project-root . --output <path>
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
