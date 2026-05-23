# Quant Research Platform Contracts and Verification

Load this reference when changing or reviewing artifact/API/frontend/A-share/Factor Factory contracts. Keep one-time run results, benchmark numbers, debug logs, and deleted temporary artifacts out of this file.

## Artifact Types

### `evaluation_set` schema `0.5`

- Source files: `src/quant_research_platform/artifacts/artifact_schema.py`, `artifact_writer.py`, `artifact_loader.py`, `manifest_validation.py`, `display_manifest.py`.
- Writer emits `artifact_schema_version="0.5"` and `artifact_type="evaluation_set"` in `manifest.json` and `summary.json`.
- Loader rejects unsupported artifact schema versions and unsupported artifact types.
- `manifest.display.schema_version` must match the current evaluation artifact schema and must be validated at API/UI entry points.
- All display tables must map to real artifact CSV paths; all download items must be backed by `manifest.display` or an explicitly allowed metadata item.
- Never reintroduce `artifacts.xlsx`, old `total_*_portfolio_return` summary fields, direct frontend assumptions about undeclared files, or fallback loaders that silently accept retired schemas.

### `factor_factory_run` schema `0.1`

- Source files: `src/quant_research_platform/research/factor_factory/reports.py`, `batch_runner.py`, `src/quant_research_platform/interfaces/api/routes/factor_factory.py`, `web/src/FactorFactoryRuns.tsx`, `web/src/api.ts`.
- Writer emits `artifact_schema_version="0.1"` and `artifact_type="factor_factory_run"`.
- API detail routes must reject wrong artifact types, wrong schema versions, missing/invalid `display`, unsafe paths, and stale manifests.
- API and frontend should expose candidate summaries through `run_ref` and `manifest.display`, not raw private paths.
- Factor Factory batch runs should write distinct factory run directories and avoid overwriting candidate reports by default.
- Candidate summaries and failed logs must sanitize private raw/interim/processed/report paths.

## API and Download Exit Criteria

- Run list, run evidence, CSV preview, direct CSV download, ZIP download, and comparison endpoints consume the same display contract.
- Multi-root evaluation APIs and frontend state must not collapse runs by `run_id` alone; `source_alias` is part of the identity.
- Direct CSV links validate display schema and exposed download items.
- ZIP item selection validates against the manifest before creating cache paths or writing temporary ZIP files.
- Uploaded-run deletion is limited to `summary.data_stage == "uploaded_panel"`, configured run roots, immediate child run directories, and directories containing `manifest.json`.
- Bad or stale runs are rejected or excluded at the earliest reasonable entry point.

## Frontend Exit Criteria

- Run switching clears stale evidence and updates title, summary cards, interpretation, charts, metric details, CSV preview, report mode, comparison state, and downloads.
- Combination switching updates every combination-dependent surface.
- Display-manifest table paths cover top-level, metrics, portfolio, diagnostics, and breakdown tables; top-level tables such as `combinations.csv` must have a preview route.
- Disabled or unavailable tabs reset to the first valid tab.
- PnL and IC / RankIC remain primary evidence charts. Turnover, single-period return, and cost diagnostics stay subordinate or switchable.
- `web/src/FactorStudio.tsx` and `web/src/PipelineBuilder.tsx` must keep A-share as the enabled market and crypto/other markets disabled or planned until real modules exist.
- Frontend creation surfaces should not imply deployment, trading, paper trading, or one-click full raw-data rebuilds.

## A-share Pipeline and Runtime Boundaries

- A-share processed data location helpers live under `src/quant_research_platform/markets/implementations/a_share/runtime/`.
- A-share private evaluation should use `python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation`.
- Use `--start-year/--end-year` to locate processed factor files and `--start-date/--end-date` for `trade_date` row filtering.
- Generic `evaluate-parquet` is only for evaluation-layer, adapter, or non-A-share local Parquet checks. It is not proof that the A-share adapter selected the right coverage file, date-filtered correctly, wrote label availability summaries, or preserved private run metadata.
- `POST /api/pipeline-runs` defaults to `execution_mode="snapshot"` and writes `run_config.yaml`, `warnings.json`, and `summary.md`.
- Backend `execution_mode="processed_existing"` calls the lightweight runner in `src/quant_research_platform/pipelines/a_share/runner.py` over existing processed factor tables. It may produce runner checkpoints, runtime summary, and evaluation manifest, but it is not full raw/interim/processed rebuild orchestration.

## Local Scripts

All local launch and verification helpers now live directly under `scripts/`:

- `scripts/check_all.ps1`
- `scripts/start_private_workbench.ps1`
- `scripts/start_api_private_runs.ps1`
- `scripts/start_web.ps1`
- `scripts/stop_workbench.ps1`
- `scripts/private_run_roots.ps1`
- `scripts/phase1_acceptance.py`

Do not document stale `scripts/local/` or `scripts/build/` command paths.

`scripts/phase1_acceptance.py` is a private Phase 1 acceptance helper. Use it only when the task intentionally covers acceptance baseline renaming, 2024 acceptance run regeneration, compact comparison, or Factor Factory smoke under `reports/private_real/`. It is not a routine check for docs-only, source-only, or public-safe work.

## Docs and Code-Map Drift Checks

When docs describe artifact, API, download, frontend, A-share pipeline, or Factor Factory contracts, verify them against code before preserving names or commands:

- Check FastAPI route coverage against `docs/code-map/platform-flow.md`; include route additions such as upload templates, delete routes, comparison, pipeline, and Factor Factory endpoints.
- Check referenced source/doc/script paths in root docs, `docs/architecture/`, and `docs/code-map/`; future-target paths must be explicitly described as planned or suggested, not current files.
- Search for retired document names, old code-map fragments, stale module paths, stale script directories, retired schema fields, and hidden frontend/backend shortcuts.
- Run Markdown link checks for root docs plus `docs/`, and use `git diff --check` before claiming the docs are synchronized.
- For docs-only work, prefer lightweight import/route checks over private acceptance runs; do not use `scripts/phase1_acceptance.py` unless private Phase 1 evidence regeneration is in scope.

## Automated Test Surfaces

For every new validation rule, add one positive path and at least one negative path when code changes are in scope. For retired behavior, prefer an absence or rejection test; keyword searches are only a backstop.

Useful surfaces:

- producer and loader: schema version, artifact type, required tables, required columns, `manifest.display` validation, path safety, and rejection of retired fields;
- API and downloads: run list, run evidence, table preview, direct CSV download, selected ZIP download, all-items ZIP download, comparison download, unknown items, unsafe paths, and cache behavior;
- uploaded-run deletion: positive uploaded-panel cleanup plus negative generated/private/missing-manifest/unknown-root/unsafe-run/out-of-root cases;
- frontend state: run identity matching, first valid combination, combination filtering, disabled tab correction, display-driven table specs, chart range readouts, chart empty states, and selected artifact URLs;
- create-run and pipeline flow: required columns, `pnl_return_column`, engine selection, `execution_mode`, time breakdowns, and user-facing validation messages;
- Factor Factory: schema `0.1`, `factor_factory_run` API validation, distinct run directories, sanitized reports, and frontend listing/detail behavior.

## Browser Acceptance

Use browser acceptance when frontend behavior changes:

- default result browser page loads and renders the selected run;
- run and combination switching update all dependent surfaces;
- full report mode expands expected evidence surfaces without internal implementation notes;
- CSV preview defaults collapsed, expands on demand, and respects preview row limits;
- all-items download starts a ZIP download without `Failed to fetch`;
- Factor Studio shows A-share enabled and crypto planned/disabled;
- Pipeline Builder validates required fields, defaults to snapshot, and does not imply deployment or trading;
- at least one narrow viewport is checked for overlap, clipped labels, and unusable controls.

After browser acceptance, stop local services and list validation outputs before cleanup. Remove only exact approved paths or single explicitly named files, and do not remove generated runs, profiles, dependencies, or private outputs as incidental cleanup.

## Skill and Memory Sedimentation

- Keep global skills for cross-project rules: Windows/Conda command forms, memory hygiene, generic skill authoring, and repository-agnostic review patterns.
- Keep this project skill for `quant-research-platform` contracts, local paths, artifact schema versions, and private-data boundaries.
- When a drift fix teaches a reusable cross-project lesson, record the recommendation in this project skill or reference first if the user did not ask for global memory. Promote it to global memory or a global skill only with explicit approval.
- Do not store transient run evidence, deleted paths, commit hashes, or temporary debug output as skill guidance.
- Prefer a lean `SKILL.md` plus linked references over a single large instruction file; add references only when they change future behavior.
