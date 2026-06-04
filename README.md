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
- Public data product id: `a_share_daily_bar`.
- Current website UI work: out of scope for this clean rewrite slice.
- Current deployment/scheduler/worker/model-registry/feature-store scope: out
  of scope.

Private boundaries:

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
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
GET  /api/0.1/catalog/universes
GET  /api/0.1/catalog/data-products
POST /api/0.1/data-products/a_share_daily_bar/interim
POST /api/0.1/markets/{market_id}/processed-data
POST /api/0.1/research-datasets
POST /api/0.1/workflows
GET  /api/0.1/reports/{run_id}
```

The clean CLI surface is:

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

Old command and endpoint names are not clean-rewrite public contracts.

## Development Checks

Use the approved Conda environment:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/architecture tests/contracts tests/interfaces/test_api_cli_01_contract.py
conda run -n analysis python -m ruff check src tests
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
