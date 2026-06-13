# Quant Research Platform

Local multi-market quantitative research platform. Current implementation phase:
QRP Clean Rewrite `0.1.0`.

This project is local research infrastructure, not investment advice, paper
trading, live trading, broker/exchange connectivity, production deployment, or
a performance guarantee. It does not log in to vendors, download vendor data,
manage credentials, or connect external market-data APIs.

## Current Boundary

- Project version: `0.1.0`.
- Public API/spec/contract version: `0.1`.
- Current enabled market module: A-share.
- Public user-facing data objects: cataloged factors, labels, and research
  dataset requirements.
- Current website UI work: out of scope for this clean rewrite slice.
- Current deployment/scheduler/worker/model-registry/feature-store scope: out
  of scope.
- Current local catalog profile composition lives in `catalog/profiles.py`.
  The public local profile builder is `local_a_share_csmar_catalogs()`,
  returning `CatalogBundle`.

Private boundaries:

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
data/research/a_share/
reports/private_real/
```

These directories are local/private. Do not commit, publish, copy, or expose raw
vendor rows, row-level processed data, private reports, credentials, secrets, or
stock-code samples.

## Source Topology

```text
src/quant_research_platform/
  contracts/        Stable 0.1 definitions and specs
  storage/          Layout, parquet, manifests, partitions, checksums, privacy
  catalog/          Queryable registries for platform objects
  sources/          Internal provider adapters
  markets/          Market semantics; first active implementation is a_share
  research/         Factors, labels, features, datasets, evaluations, models,
                    predictions, portfolios
  workflows/        Simple DAG specs, planner, executor, checkpoints
  compute/          DataFrame/tensor compute backends
  agents/           Proposal-only Factor/Label/Dataset/Workflow boundaries
  interfaces/       CLI and local API adapters
  reports/          Report manifests, summaries, and bundles
  runtime/          Privacy, sanitization, logging, profiling, errors
```

Retired compatibility structure is not the target contract. Its retained
responsibilities move under the domains above.

## API And CLI

The clean API uses `/api/0.1/...`, without a `v` prefix:

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

The clean CLI surface is:

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

These public routes are read-only. Source/interim/processed materialization and
private real-data execution are internal workflows and are not exposed through
the API/CLI.

Removed public placeholders are not compatibility requirements. Future-only
work remains explicit: `processed-data-runs`, `workflow-runs`, `report
index/show`, `RunRepository`, `ReportRepository`, and `private plugin`.

Private local-data CLI:

```text
qrp-private schema-inventory --project-root . --output <path>
```

`qrp-private` currently exposes only sanitized schema inventory. It does not run
private import, private rebuild, or private evidence regeneration. Treat this as
private aggregate schema inventory, not as public-safe aggregate evidence,
because logical source paths or source field names may be included.

## Development Checks

The environment contract is documented in
`docs/architecture/environment-contract.md`.

Create or update the project Conda environment from `environment.yml`:

```powershell
conda env update -f environment.yml --prune
```

Then run checks:

```powershell
$env:PYTHONPATH="src"
conda run -n quantlab313 python -m pytest -q tests/architecture tests/contracts tests/interfaces/test_api_cli_01_contract.py
conda run -n quantlab313 python -m ruff check src tests
git diff --check
```

Run broader tests only when the active slice owns the affected surfaces.

## Documentation Map

- `AGENTS.md`: project execution boundaries and stop conditions.
- `PROJECT_STATUS.md`: current operational status and handoff signal.
- `docs/api/README.md`: clean API/CLI contract.
- `docs/code-map/`: implementation maps.
- `docs/data/`: public-safe data layout notes.
- `.agents/skills/quant-research-platform-maintenance/`: Codex maintenance
  workflow, routing, and verification references.
