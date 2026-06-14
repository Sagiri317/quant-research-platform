# Quant Research Platform 项目状态

P0/P1/P2/P3 architecture and rebuild follow-ups are tracked in
`docs/architecture/open-issues-register.md`.

本文只记录当前 operational truth、当前 blocker / next steps、以及需要同步到
workspace root 的 handoff 信号。历史 closeout、旧分支叙事、私有 run 细节和
一次性调试证据不写入本文。

## 当前状态

| 项目 | 当前事实 |
|---|---|
| 当前阶段 | QRP Clean Rewrite `0.1.0` on Local Research Platform |
| 当前分支 | `feature/qrp-clean-rewrite` |
| 版本边界 | project `0.1.0`; API/contracts use `0.1` |
| 源码拓扑 | `contracts`, `storage`, `catalog`, `sources`, `markets`, `research`, `workflows`, `compute`, `agents`, `interfaces`, `reports`, `runtime` |
| 当前 API/CLI | `/api/0.1/...` factor/label catalog routes and research dataset requirements; `qrp catalog factors`, `qrp catalog labels`, `qrp research-datasets requirements` |
| 执行边界 | API/CLI is read-only for current public users; source/interim/processed materialization and private real-data execution are internal |
| 市场范围 | A-share enabled; other markets are future design only, with no empty implementation packages |
| 私有边界 | `data/raw/<internal-provider>`, `data/interim/<internal-provider>`, `data/processed/a_share`, `data/research/a_share`, `reports/private_real` |
| 交易边界 | No investment advice, paper/live trading, broker/exchange connector, deployment, scheduler, worker, or external data API |

## 当前有效样例

Current clean-rewrite verification is code/test based. Private generated
artifacts remain local evidence only. The previous public-safe aggregate parity
and coverage evidence in `docs/reports/` is
`superseded_by_factor_catalog_refactor` and
`evidence_pending_regeneration` until the next private rebuild.

Formal verification uses Python `>=3.13,<3.14` with conda `quantlab313`.
The primary verified OS is Windows 11. macOS/Linux are cross-platform
compatible targets, but PASS must only be claimed after tests run there.

Current public-safe anchors:

- 0.1 source topology tests.
- 0.1 contracts/storage/catalog/source tests.
- 0.1 A-share processed, ResearchDataset, evaluation, model, and prediction
  tests.
- A-share daily price-volume factors and `forward_return` labels use
  per-asset valid daily bars, not calendar sessions.
- 0.1 API/CLI route and command contract tests.
- Public-safe parity-or-explained-delta evidence for processed panel, factor
  values, labels, masks/universes, ResearchDataset, evaluation, and prediction
  smoke is superseded pending regeneration after the factor catalog refactor.
- P2 compute portfolio and internal provider contract split modules use direct
  owner modules only.
- Report writers derive paths through `ReportSpec`, `ReportCatalog`,
  `StorageLayout`, and `run_id`.
- Prior local verification covered the new processed factor/label-supporting
  layout, ResearchDataset layout, and full historical `daily_panel` aggregate
  coverage for `2014-01-01` through `2025-12-31` without committing row-level
  output artifacts; full public-safe evidence/report regeneration remains
  pending after the factor catalog refactor.

## 当前 blocker / 风险

1. This branch is not merged to `main`.
2. Future-only items remain out of the active public contract:
   `processed-data-runs`, `workflow-runs`, `report index/show`,
   `RunRepository`, `ReportRepository`, and `private plugin`.
3. Active private local-data CLI is limited to
   `qrp-private schema-inventory --project-root . --output <path>`; private
   import, rebuild, and evidence regeneration remain out of scope.

## 下一步

1. Complete the data-layer residual closure and handoff package verification.
2. Commit and push `feature/qrp-clean-rewrite` only after final verification
   passes.
3. Do not merge `main` from this task.

## Root Handoff / 根目录回传信号

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-06-03 | status | QRP clean rewrite started on `feature/qrp-clean-rewrite`: source topology moved toward 0.1.0, API/CLI public surface changed to `/api/0.1/...`, private data boundaries changed to `data/raw`, `data/interim`, `data/processed`, and `reports/private_real`; old endpoint/command compatibility is no longer target-contract truth. | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-04 | status | QRP clean rewrite completed through Phase 10 locally: 2014 parity-or-explained-delta passed, 2014-2025 factor/label-supporting processed layers and ResearchDataset were migrated through spec-driven writers, and reports remain under `reports/private_real`. | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-04 | status | Pre-merge fix added public-safe clean rewrite parity evidence, thin API/CLI dry-run validation services, sanitizer/report/writer gates, and full historical `daily_panel` aggregate coverage for 2014-2025 without committed row-level output. | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-04 | status | Pre-merge blocker repair updated `execution_mode`, registered workflow execution, ReportSpec/ReportCatalog writer gates, P2 split modules, and public-safe aggregate evidence hash validation. | WORKSPACE_STATUS, PLAN_INDEX | open |

## First-Stage Data-Layer Closure

Status: first-stage data-layer closure is implemented for the A-share
internal-provider profile.

The closed first-stage path is:

```text
source/raw -> interim -> A-share processed facts -> A-share annotations/masks/context -> factor data -> label data -> research dataset
```

`daily_panel` is the A-share fact layer. It does not own derived research or
execution masks such as `eligible`, `can_enter`, or `can_exit`.

A-share trading semantics are owned by
`markets/implementations/a_share/processed/annotations.py`. Eligibility,
as-of, execution, and evaluation annotation datasets are derived from
`daily_panel` facts and are consumed downstream rather than recomputed by label,
evaluation, model, or prediction code.

The generic research dataset assembler is market-agnostic. It uses
`ResearchDatasetSpec.base_input_data_id` as the explicit base input, reads that
input first, left-joins the remaining inputs on the primary key, and emits
columns in primary-key, feature, label, filter, context order. Non-base inputs
use strict key coverage by default: they must cover the base primary-key set,
value-column nulls are allowed, and keys outside the base fail closed. Sparse
inputs are allowed only through explicit `key_coverage` opt-in and still fail
closed on keys outside the base. `ResearchDatasetSpec.key_coverage` is
normalized for every declared input and persisted into research dataset
`spec.json` and `_manifest/manifest.json`; the base input is always `strict`.

A-share processed annotation, factor, and label materializers resolve
`ProcessedDataSpec.processed_input_data_ids` through internal input contracts.
Current `price_volume` factors and `forward_return` labels use `daily_panel` as
their primary processed input by contract, without adding public factor or label
registries.

Evaluation, model, and prediction catalogs remain research profile demo
components and are outside the first-stage data-layer closure. Evaluation
diagnostics, model registry, prediction registry, workflow execute/resume,
frontend/dashboard work, high-frequency, multi-market implementation, and
paper/live trading remain out of scope.

The current research profile composition root is `catalog/profiles.py`.
The public research profile composition root is `research_profile()`, returning
`ResearchProfile`.

Full-horizon private reruns are not required for this first-stage structural
closure unless a run artifact explicitly records that such a rerun occurred.
