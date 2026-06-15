# Contracts and Verification / 合同与验证

本文件是高变动 contract map（合同地图）。具体行为以代码、tests 和 schema/manifest contracts 为准；本文用于路由和 gates。

## Versions

- Project version: `0.1.0`.
- API/spec/contract version: `0.1`.
- Python: `>=3.13,<3.14`.
- Environment contract: `docs/architecture/environment-contract.md`.

## Public API / CLI contract

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements

qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

Future-only public items are not active contracts: `processed-data-runs`, `workflow-runs`, `report index/show`, `RunRepository`, `ReportRepository`, `private plugin`, universal artifact lifecycle API, model registry, feature store.

Private local CLI is narrow:

```text
qrp-private schema-inventory --project-root . --output <path>
```

## Private roots

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
data/research/a_share/
reports/private_real/
```

Public-safe outputs must not include row-level private data, raw vendor exports, stock-code samples, local absolute paths, vendor markers, credentials, tokens, or raw tracebacks with local/private context.

## Artifact truth statuses

Use only these meanings:

| Status | Meaning |
|---|---|
| `PASS` | Check ran and passed. |
| `FAIL` | Check ran and failed. Keep failure visible. |
| `NOT_RUN` | Check did not run, or evidence is absent. |
| `NOT_APPLICABLE` | Check is structurally irrelevant to this task. |
| `BLOCKED` | Check was required but blocked by environment, dependency, approval, or missing tool. |

Forbidden: marking skipped checks as `PASS`; converting official compare `FAIL` to `PASS`; presenting smoke-scoped evaluation/prediction evidence as full-horizon evidence.

## Verification ladder

1. Pre-edit: `git status --short --branch`; classify task mode and owner layer.
2. Red test / expectation: when changing behavior, identify the failing or missing test/guard before implementation.
3. Narrow checks: direct unit/contract tests for touched owner layer.
4. Same issue family scan: `rg` same-family terms across owner layer and direct dependents.
5. Public surface gate: ensure API/CLI/schema/report/display surfaces did not expand unless approved.
6. Private safety gate: no private data, blocked file types, or local paths in outputs.
7. Docs/status sync gate: update only active docs that reflect changed current truth.
8. Final truthfulness gate: report run/not-run/fail status exactly.

## Common commands

Use the commands documented in `README.md` and task-specific tests. For governance-only changes, use:

```text
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only
python .agents/skills/quant-research-platform-maintenance/scripts/package_safety_check.py <zip-or-dir>
git diff --check
```

Only claim OS/platform PASS after running on that OS/platform.
