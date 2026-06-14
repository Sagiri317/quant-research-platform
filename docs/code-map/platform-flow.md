# Platform Flow Code Map

This map records the QRP Clean Rewrite `0.1.0` implementation flow. It is a
current code-reading map, not a roadmap or historical closeout.

## Flow

```text
internal source adapters -> source-normalized interim artifacts
-> markets/implementations/a_share session-windowed processed materializer
-> data/processed/a_share/{panel,annotation,factor,label}
-> research/datasets session-chunked assembly
-> data/research/a_share
-> research/evaluations, research/models, and research/predictions
-> reports/private_real
-> interfaces/api and interfaces/cli read research/catalog surfaces
```

## Package Ownership

| Package | Responsibility |
|---|---|
| `contracts/` | Stable 0.1 definitions and specs |
| `storage/` | Layout, parquet, manifests, year partitions, checksums, privacy |
| `catalog/` | Queryable source of truth for platform objects |
| `sources/` | Internal provider adapters |
| `markets/` | Market-specific semantics, first active market: `a_share` |
| `research/` | Factors, labels, features, session-chunked datasets, evaluations, models, predictions, portfolios |
| `workflows/` | Simple DAG specs, planner, executor, checkpoints |
| `compute/` | DataFrame and tensor compute backends |
| `agents/` | Proposal-only factory/planner boundaries |
| `interfaces/` | Local CLI and API adapters |
| `reports/` | Report catalog and private report artifact writers |
| `runtime/` | Privacy, sanitization, logging, profiling, errors, resource chunk planning |

## Artifact Ownership

- Dataset instance metadata is owned by `storage/datasets.py`.
  `write_dataset(instance_metadata=...)` accepts only explicit, guarded
  metadata keys such as `input_fingerprints`; research dataset assembly does not
  mutate `_manifest/manifest.json` after writing.
- Evaluation owns `EvaluationDiagnostics`, IC summary, and group-spread summary
  construction. Reports validate and persist evaluation frames as private
  artifacts without importing compute. Evaluation summary statistics use
  explicit valid counts, and no-valid IC or RankIC values remain `NaN`.
- Prediction/model handoff uses fixed-shape `model_metadata` and
  `prediction_metadata` report manifest payloads. Prediction metadata records
  the effective train/test window values, including `train_end`.
- Evaluation, model fitting, and prediction generation use the shared
  `runtime.resources` policy for internal timestamp-session chunking when the
  resource budget requires it. These are private implementation details and do
  not add public run APIs.
- No model registry, prediction registry, universal artifact ID, public run API,
  or report lookup is part of `0.1.0`.

## API Flow

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

The API is local-only. Catalog routes return registered factor and label
metadata. The research dataset requirements route is read-only and does not
build artifacts. Source/interim/processed materialization is internal. The API
does not
expose private real-data execution, deployment, scheduler, worker, paper/live
trading, external data, broker/exchange, or website UI contracts.

## CLI Flow

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

No private-source root flags, materialization commands, or overwrite/force
controls are part of the `0.1.0` CLI contract.

Retired commands and endpoints are not current clean-rewrite contracts.

Future-only items remain outside the active surface: `processed-data-runs`,
`workflow-runs`, `report index/show`, `RunRepository`, `ReportRepository`,
and `private plugin`.

Private local schema inventory is exposed separately through
`qrp-private schema-inventory --project-root . --output <path>`. It scans only
sealed dataset roots and emits private aggregate schema inventory metadata. This
inventory may include logical source paths or source field names, so it is not
the same thing as public-safe aggregate evidence.

The existing private verification evidence harness is exposed through
`qrp-private evidence run` for local smoke/evidence generation only. It supports
optional internal evaluation and prediction window overrides, but this does not
create public `evaluate`, `predict`, or `report` commands.

## Narrow Follow-up Notes

- Workflow summaries currently expose `completed_nodes`; a future summary model
  may separate `planned_nodes` from `completed_nodes` so dry-run plans are not
  represented as completed work.
- Storage writer root validation currently validates spec-shaped legal suffixes;
  a future implementation should make this project-root-relative, likely by
  passing `StorageLayout.project_root` or equivalent through writer validation
  paths.
