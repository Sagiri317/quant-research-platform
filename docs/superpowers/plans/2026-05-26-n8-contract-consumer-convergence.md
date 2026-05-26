# N8 Contract Consumer Convergence Implementation Plan

Date: 2026-05-26

Branch: `codex/contract-consumer-convergence`

Status: planned. This plan is written on `main`; implementation should start from a clean `main` branch after the plan is committed.

## Goal

Converge the stable artifact/display/API/frontend consumption contracts before N5 risk/regime diagnostics and N6 prediction handoff add more display tables. The work should reduce duplicated path parsing, frontend route inference, mixed API responsibilities, and legacy/dev upload ambiguity without changing research calculations.

N8 is an engineering-contract branch. It must not recalculate eligibility, tradability, IC, returns, weights, turnover, risk diagnostics, or prediction outputs. It must not expose private CSMAR details, connect external APIs, add deployment capability, or change trading boundaries.

## Scope

Implement these four ordered slices:

1. `manifest.display` table preview/download resolver and validation tightening.
2. Evidence API table preview metadata and generic display-table preview route.
3. Frontend metadata consumer plus CSV upload legacy/dev demotion.
4. Factor-template route/service split plus A-share evaluation service entry.

The implementation should stay inside the existing source tree:

- `src/quant_research_platform/artifacts/`
- `src/quant_research_platform/interfaces/api/`
- `src/quant_research_platform/markets/implementations/a_share/evaluation/`
- `src/quant_research_platform/pipelines/a_share/`
- `src/quant_research_platform/research/factor_factory/`
- `tests/`
- `web/src/features/`
- `web/src/shared/`
- `docs/`

## Non-Goals

- Do not redesign the Evidence UI visual system.
- Do not change A-share `start_from` semantics.
- Do not add N5 risk/regime artifacts or N6 prediction artifacts.
- Do not add external vendor APIs, broker APIs, paper trading, live trading, order routing, deployment, or production automation.
- Do not create repo-level `apps/api` or empty future market/app shells.
- Do not bump artifact schema to `0.7` unless a later implementation finding proves a persisted artifact structure change cannot be represented as a schema `0.6` display-contract tightening.
- Do not delete CSV upload support in this branch. Keep the backend compatibility path and demote the frontend/docs semantics.

## Current Findings

- Evaluation-set artifacts already use schema `0.6` and `manifest.display.tables` with `key`, `path`, `format`, `download_item`, `join_key`, `preview_rows`, and `supports_combination_filter`.
- API preview routes still expose path-family endpoints: `metrics`, `portfolio`, `diagnostics`, and `breakdowns`.
- The frontend currently derives preview routes from artifact paths in `web/src/shared/api/api.ts` via `tableApiPath()`. This duplicates backend routing rules and requires new frontend hardcoding for every future display-table family.
- N2 diagnostics latest-row display uses the preview API correctly, but the URL still comes from frontend path inference.
- Factor Factory route code has its own display table/download path validation and CSV preview logic, separate from evaluation-set artifact package helpers.
- `GET /api/factor-templates` currently lives in `interfaces/api/routes/pipeline.py` and `interfaces/api/services/pipeline_service.py`, even though template discovery is not a pipeline snapshot/run responsibility.
- CSV upload remains useful as a local uploaded-panel smoke/compatibility path, but it should not read like the primary A-share run creation path.
- A-share CLI and pipeline runner call evaluation code through CLI-like or adapter-level seams rather than one explicitly named service entry.

## Target Contract

### Artifact Manifest

Keep `manifest.json` portable and artifact-local. Do not write `source_alias`, `run_id`, absolute URLs, hostnames, or API base paths into generated artifacts.

Each `manifest.display.tables[]` entry should continue to declare artifact identity and add a portable preview descriptor:

```json
{
  "key": "tradability_diagnostics",
  "label": "可交易性诊断",
  "group": "diagnostics",
  "role": "diagnostics",
  "path": "diagnostics/tradability_diagnostics.csv",
  "format": "csv",
  "download_item": "diagnostics/tradability_diagnostics.csv",
  "dimension_source": "dimensions/combinations.csv",
  "join_key": "combination_id",
  "preview_rows": 20,
  "supports_combination_filter": true,
  "preview": {
    "kind": "table",
    "table_key": "tradability_diagnostics",
    "default_limit": 20,
    "max_limit": 1000,
    "supports_offset": true,
    "supports_combination_filter": true,
    "filter_parameter": "combination_id"
  }
}
```

For non-combination tables, `filter_parameter` should be `null` and `supports_combination_filter` should be `false`.

This is a schema `0.6` display-contract tightening. Existing generated schema `0.6` runs that lack the required `preview` descriptor become stale generated runs for current showcase/testing purposes. Do not weaken the contract to keep stale generated runs visible; regenerate generated runs with approval when they are needed for current acceptance.

### Evidence API

The API should enrich the portable display table metadata with run-specific preview URLs at response time. The runtime evidence payload may add:

```json
{
  "preview_url": "/api/runs/{source_alias}/{run_id}/tables/{table_key}/preview"
}
```

The API should also expose one generic metadata-driven route:

```text
GET /api/runs/{source_alias}/{run_id}/tables/{table_key}/preview
```

Query parameters:

- `limit`: required or defaulted from `manifest.display.tables[].preview.default_limit`.
- `offset`: optional, default `0`.
- `combination_id`: optional, honored only when the display table declares `supports_combination_filter=true`.

Existing path-family preview endpoints should remain compatible during N8, but delegate to the same shared resolver. The frontend should stop using them directly.

### Frontend

The frontend should consume `RunDisplayTable.preview_url` or equivalent runtime metadata. It should stop deriving preview endpoints from `table.path`.

`table.path` remains the artifact path for display/download context only. `download_item` remains the direct download contract.

Combination-aware refresh should be metadata-driven:

- chart tables refresh when a chart references the table and the table supports `combination_id`;
- diagnostics and future N5/N6 combination-aware tables refresh when table metadata declares the same capability;
- the frontend should not maintain a growing set of hardcoded table keys such as `tradability_diagnostics`.

### CSV Preview Semantics

Preview is an artifact inspection/audit support feature. It is not the primary analysis UI, not a generic spreadsheet viewer, and not a prediction preview.

Keep the UI small:

- default 20 rows;
- no full-table load;
- no generic search/sort/pagination expansion in N8;
- use table metadata for the endpoint and limits;
- full data access stays through manifest-approved downloads.

## Slice 1: Display Resolver And Validation

Files to inspect/update:

- `src/quant_research_platform/artifacts/display_manifest.py`
- `src/quant_research_platform/artifacts/manifest_validation.py`
- `src/quant_research_platform/artifacts/artifact_package.py`
- possible new module: `src/quant_research_platform/artifacts/display_tables.py`
- `tests/artifacts/test_schema_06_contract.py`
- related artifact package tests

Implementation steps:

1. Add a small display-table resolver module with typed functions for:
   - finding a display table by `key`;
   - resolving a display table path against `artifacts`;
   - reading `preview_rows` / `preview` limits;
   - checking `download_item` consistency;
   - preserving schema-specific validation callers.
2. Add `preview` descriptor generation in `display_manifest._table_entry()`.
3. Tighten `validate_display_manifest()`:
   - every display table must have a valid `preview` object;
   - `preview.table_key` must equal `table.key`;
   - `preview.default_limit` must equal or be compatible with `preview_rows`;
   - `preview.max_limit` must be between default limit and `DISPLAY_PREVIEW_ROWS_MAX`;
   - `preview.supports_combination_filter` must match `supports_combination_filter`;
   - `preview.filter_parameter` must be `combination_id` only for combination-filterable tables, otherwise `None`.
4. Keep artifact schema version at `0.6`.
5. Add tests that fail on missing/stale preview metadata and pass on writer-generated display metadata.
6. Keep `artifact_package.py` as the download/ZIP authority. If a new resolver is shared, `artifact_package.py` should call it rather than duplicate display traversal.

TDD checks for this slice:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts/test_schema_06_contract.py tests/artifacts
```

## Slice 2: Evidence API Preview Metadata

Files to inspect/update:

- `src/quant_research_platform/interfaces/api/routes/runs.py`
- `src/quant_research_platform/interfaces/api/payloads.py`
- possible new module: `src/quant_research_platform/interfaces/api/services/table_preview_service.py`
- `tests/interfaces/api/test_api.py`
- `tests/interfaces/api/test_route_modules.py`
- `docs/code-map/evaluation-artifacts-api.md`

Implementation steps:

1. Move table preview loading into an API service that accepts `(run_dir, manifest, table_key, limit, offset, combination_id)`.
2. Add the generic route:
   - `/api/runs/{source_alias}/{run_id}/tables/{table_key}/preview`
   - resolves only `manifest.display.tables[]` keys;
   - validates requested table is display-exposed;
   - applies descriptor limits and filter support.
3. Keep existing `metrics`, `portfolio`, `diagnostics`, and `breakdowns` preview routes as compatibility wrappers that resolve their legacy path mapping and delegate to the same service.
4. In `run_evidence_payload()` or a nearby API serializer, enrich every display table with a runtime `preview_url`.
5. Preserve `(source_alias, run_id)` identity in all generated URLs.
6. Ensure direct downloads and selected ZIP still use `artifact_package.py`, not the preview service.
7. Add API tests for:
   - evidence payload includes `preview_url`;
   - generic route returns the same rows as the old diagnostics route for `tradability_diagnostics`;
   - `limit` above max is rejected;
   - `offset` works;
   - `combination_id` is rejected or ignored according to descriptor policy for non-filterable tables;
   - stale display without `preview` fails validation through the normal manifest path.

TDD checks for this slice:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/interfaces/api/test_api.py tests/interfaces/api/test_route_modules.py
```

## Slice 3: Frontend Metadata Consumer And Upload Demotion

Files to inspect/update:

- `web/src/shared/api/api.ts`
- `web/src/shared/testFixtures.ts`
- `web/src/features/evidence/useRunEvidence.ts`
- `web/src/features/evidence/useTablePreview.ts`
- `web/src/features/evidence/CsvBrowser.tsx`
- `web/src/features/evidence/TradabilityDiagnostics.tsx`
- `web/src/features/evidence/runEvidenceState.ts`
- `web/src/features/evidence/*.test.tsx`
- `web/src/features/uploads/`
- `web/src/shell/App.tsx`
- `web/src/shell/App.test.tsx` only for navigation/shell assertions that cannot be feature-local

Implementation steps:

1. Extend shared API types for runtime table preview metadata:
   - `RunDisplayTable.preview`;
   - `RunDisplayTable.preview_url` or `RunTableSpec.previewUrl`;
   - keep `path` as artifact path.
2. Remove frontend route inference from `tableApiPath()`. The resulting table spec should fail fast if a display table lacks a runtime preview URL.
3. Update `fetchTablePreview()` to accept a preview URL rather than a path-derived route.
4. Replace hardcoded combination preview table-key sets with metadata filtering.
5. Keep latest diagnostics behavior efficient:
   - first request uses default preview limit;
   - latest-row request uses `limit=1&offset=total_rows-1` only after the preview payload reports an exact enough `total_rows`;
   - do not raise default preview limits to 1000.
6. Keep CSV Browser as an audit preview:
   - no generic full viewer features;
   - label copy as "artifact preview" / "明细预览";
   - keep full data access through manifest-approved direct download.
7. Demote CSV upload UI:
   - keep backend `POST /api/runs` compatible;
   - label frontend entry as local dev/legacy uploaded-panel smoke path;
   - avoid implying it creates official A-share private runs;
   - do not add new upload features.
8. Move new tests into `web/src/features/evidence/` or upload feature tests where possible. Do not further expand broad shell-level `App.test.tsx` except for shell-only behavior.

Frontend checks:

```powershell
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
```

## Slice 4: Factor Templates And A-Share Service Entry

Files to inspect/update:

- `src/quant_research_platform/interfaces/api/routes/pipeline.py`
- `src/quant_research_platform/interfaces/api/services/pipeline_service.py`
- new route: `src/quant_research_platform/interfaces/api/routes/factor_templates.py`
- new service: `src/quant_research_platform/interfaces/api/services/factor_template_service.py`
- `src/quant_research_platform/interfaces/api/app.py`
- `src/quant_research_platform/markets/implementations/a_share/evaluation/run_factor_evaluation.py`
- new or updated service under `src/quant_research_platform/markets/implementations/a_share/evaluation/`
- `src/quant_research_platform/pipelines/a_share/runner.py`
- `src/quant_research_platform/research/factor_factory/batch_runner.py`
- tests under `tests/interfaces/api/`, `tests/pipelines/a_share/`, and `tests/research/factor_factory/`

Implementation steps:

1. Move factor-template payload models and `list_factor_template_payloads()` out of pipeline service into a factor-template service.
2. Move `GET /api/factor-templates` out of pipeline route into `routes/factor_templates.py`.
3. Include the new route in `app.py`. Preserve the public endpoint path and payload shape.
4. Keep pipeline service focused on snapshot/run payloads, checkpoint/status, and pipeline validation/execution.
5. Introduce an A-share evaluation service entry for processed factor evaluation:
   - accepts structured parameters instead of CLI `argv`;
   - calls existing `evaluation_adapter.run_processed_factor_evaluation`;
   - returns manifest path and run records;
   - keeps `run_factor_evaluation.py` as a thin CLI wrapper.
6. Update pipeline runner and Factor Factory callers only if they currently bypass or duplicate evaluation orchestration. Do not change calculations or `start_from` behavior.
7. Add tests that prove:
   - `GET /api/factor-templates?market=a_share` payload remains unchanged;
   - pipeline snapshot/run tests still pass;
   - A-share CLI wrapper delegates to the service;
   - Factor Factory template/evaluation behavior remains compatible.

Backend checks for this slice:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/interfaces/api/test_factor_pipeline.py tests/interfaces/api/test_pipeline_service.py tests/pipelines/a_share tests/research/factor_factory
```

## Documentation Updates

Update these after implementation:

- `docs/architecture/next-stage-roadmap.md`: mark N8 details completed or adjust residual work.
- `docs/architecture/next-stage-worktree-plan.md`: update branch status and follow-on conflicts.
- `docs/code-map/evaluation-artifacts-api.md`: reflect generic display-table preview route, runtime preview URLs, resolver ownership, Factor Factory sharing, factor-template route split, and CSV upload legacy/dev role.
- `docs/code-map/platform-flow.md` and `docs/code-map/platform-flow-interactive.html`: update nodes/edges if route ownership or API relationships changed.
- `PROJECT_STATUS.md`: add an open signal if N8 changes current topic, phase, next step, data/API/Git/Conda/privacy boundary, or handoff status.
- `.codex/skills/quant-research-platform-maintenance/SKILL.md`: update only if N8 discovers a reusable maintenance rule, not for one-time progress.

## Full Verification

Use narrow tests while developing each slice, then run the full N8 gate before review:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts tests/interfaces/api tests/pipelines/a_share tests/research/factor_factory
conda run -n analysis python -m ruff check src tests scripts
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
git diff --check
```

If API payload or frontend behavior changes, run browser acceptance against FastAPI + Vite:

```powershell
.\scripts\start_private_workbench.ps1
$env:CODEX_HOME="C:\Users\DELL\.agents"
& scripts\playwright_cli_project.ps1 open http://127.0.0.1:5173
& scripts\playwright_cli_project.ps1 snapshot
& scripts\playwright_cli_project.ps1 screenshot
```

Do not count an API-offline `Failed to fetch` state as acceptance evidence for this branch.

## Stop Rules

Stop and reassess before editing further if any of these happen:

- A required change would expose, print, or transform private CSMAR raw/interim/processed rows.
- A required change would alter eligibility, tradability, IC, return, weight, turnover, factor, risk, or prediction calculations.
- The cleanest implementation requires a public API breaking change beyond adding metadata/generic routes while preserving existing endpoints.
- A schema bump to `0.7` appears necessary.
- CSV upload removal looks necessary.
- The branch starts turning into a visual redesign or N5/N6 research feature branch.

## Implementation Order

1. Create `codex/contract-consumer-convergence` from clean `main`.
2. Implement Slice 1 and commit.
3. Implement Slice 2 and commit.
4. Implement Slice 3 and commit.
5. Implement Slice 4 and commit.
6. Sync docs/code maps/status.
7. Run full verification and browser acceptance.
8. Review for hardcoded frontend route inference, stale generated-run accommodation, private-data leakage, and accidental N5/N6 scope creep.
