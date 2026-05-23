---
name: quant-research-platform-maintenance
description: Use when changing quant-research-platform artifact schema, manifest.display, evidence API, download packaging, React evidence UI, A-share pipeline runs, generated runs, or docs that describe those contracts.
---

# Quant Research Platform Maintenance

## Purpose

Keep generated evaluation artifacts as the source of truth, `manifest.display` as the UI/download contract, and the React evidence workbench as a consumer of that contract. Do not restore old schema fallbacks or hidden frontend business logic.

Project `AGENTS.md` owns hard safety boundaries: local-only scope, Conda environment, data/API/secrets limits, forbidden operations, private CSMAR boundaries, and high-level code ownership. This skill owns the operational workflow for artifact contract, evidence API, download packaging, frontend evidence UI, generated runs, and docs/code-map plus architecture-doc synchronization.

## Entry Points

Start by reading project `AGENTS.md`, then inspect only the files touched by the requested surface.

| Surface | Start here |
| --- | --- |
| Artifact schema and loader | `src/quant_research_platform/artifacts/artifact_writer.py`, `artifact_loader.py`, `artifact_schema.py` |
| Display contract | `src/quant_research_platform/artifacts/display_manifest.py` |
| Evidence API and downloads | `src/quant_research_platform/interfaces/api/routes/`, `interfaces/api/services/`, `interfaces/api/app.py`, `interfaces/api/run_index.py` |
| A-share pipeline runs | `src/quant_research_platform/pipelines/a_share/`, `config/a_share/pipelines/` |
| A-share private evaluation | `src/quant_research_platform/markets/implementations/a_share/evaluation/run_factor_evaluation.py`, `evaluation/evaluation_adapter.py`, `runtime/paths.py` |
| Factor Factory | `src/quant_research_platform/research/factor_factory/`, `src/quant_research_platform/interfaces/api/routes/factor_factory.py`, `tests/research/factor_factory/`, `tests/interfaces/api/test_factor_factory_api.py` |
| Frontend evidence UI | `web/src/runEvidenceState.ts`, `ResultsView.tsx`, `RunCharts.tsx`, `ArtifactDownloads.tsx`, `CsvBrowser.tsx` |
| Docs and maps | `README.md`, `ARCHITECTURE.md`, `EVALUATION_INTERFACE.md`, `RESEARCH_CARD.md`, `PROJECT_STATUS.md`, `docs/architecture/`, `docs/code-map/` |

Use PowerShell `Select-String` when `rg` is unavailable or denied.

## Contract Workflow

For broad changes, stabilize the artifact contract before changing the UI. Do not let the page design become the source of truth.

1. Define the artifact schema, table names, column names, `combination_id` semantics, and `manifest.display` tables/downloads/presets/charts first.
2. Update writer output, loader validation, lightweight run index validation, API payload/download behavior, frontend consumers, tests, docs, and code maps in the same round.
3. Remove legacy compatibility in one pass when old generated runs are no longer supported. Delete fallback logic from loader, run index, comparison, inventory, tests, docs, and frontend mocks together.
4. Wire user-facing artifacts through `manifest.display`; do not add backend-only, frontend-only, or direct-link-only files.
5. Preserve API-visible run identity as `(source_alias, run_id)` across run list, evidence routes, CSV preview, direct downloads, ZIP cache paths, comparison payloads, and frontend route state.
6. Review the diff by surface: core artifact contract, API/download, frontend UI, A-share pipeline, tests/fixtures, docs/skills/gitignore, and generated or temporary outputs.

Current artifact schema is `0.5`. Treat unsupported or legacy artifact schemas as regeneration candidates, not compatibility targets, unless the user explicitly asks to reintroduce compatibility.

Never reintroduce:

- `artifacts.xlsx`;
- old `total_*_portfolio_return` summary fields;
- direct frontend assumptions about files that are not in `manifest.display`;
- broad fallback loaders that silently accept retired schemas.

## Contract Exit Criteria

Do not call a broad artifact/API/frontend refactor complete until every gate below has been checked. Browser usability is necessary but not sufficient.

### Producer and Loader

- The writer emits exactly one current schema version.
- The loader rejects unsupported schema versions instead of silently upgrading or falling back.
- `manifest.display.schema_version` is validated wherever a run can enter the UI/API surface, including lightweight run index loading.
- Every artifact CSV path is represented in `manifest.display.tables`.
- Every display table has a matching download item, and unknown download items are either known metadata items or explicitly rejected.
- No old summary fields, old table names, or old compatibility branches remain unless the user explicitly requested backward compatibility.

### API and Download

- Run list, run evidence, CSV preview, direct CSV download, ZIP download, and comparison endpoints all consume the same contract.
- Multi-root APIs and frontend state must not collapse runs by `run_id` alone; `source_alias` is part of the identity.
- Direct CSV links validate display schema and exposed download items.
- ZIP item selection validates against the manifest before creating cache paths or writing temporary ZIP files.
- Uploaded-run deletion must be tightly gated: allow only `summary.data_stage == "uploaded_panel"`, resolve the run through configured run roots, require the run directory to be an immediate child of the selected root, require `manifest.json` inside the run directory, and reject all non-uploaded or out-of-root paths.
- Bad or stale runs are rejected or excluded at the earliest reasonable entry point, not after the frontend has already selected them.

### Frontend State

- Run switching cannot show stale evidence from the previous run.
- Combination switching updates summary cards, interpretation, metric details, charts, CSV preview, downloads, report mode, and comparison state where applicable.
- Display-manifest table paths cover top-level, metrics, portfolio, diagnostics, and breakdown tables; top-level tables such as `combinations.csv` must have a valid preview route.
- Disabled or unavailable tabs are automatically reset to the first valid tab.
- PnL and IC / RankIC remain primary evidence charts. Turnover, single-period return, and cost diagnostics stay visually subordinate or switchable.
- Factor Studio and Pipeline Builder must keep A-share enabled and crypto/other markets disabled or planned until real modules exist.

### Tests and Fixtures

- Tests fail for stale schema/display versions at the same entry points the app exposes.
- Tests cover absence of retired behavior, not only presence of new behavior.
- Frontend fixtures either come from a checked backend fixture snapshot or are intentionally kept in sync with `manifest.display` rules.
- New validation rules have one positive path and at least one negative test.
- A-share pipeline snapshots record config, warnings, and summary without overwriting existing private runs by default.
- A-share real-data acceptance uses `python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation` after the approved structural migration; use generic `evaluate-parquet` only for evaluation-layer or adapter checks, not as a substitute for A-share adapter validation.
- Date-filtered A-share runs should exercise both `--start-year/--end-year` coverage-file selection and `--start-date/--end-date` row filtering when that behavior is in scope.
- Factor Factory writes each batch into a distinct factory run directory, does not overwrite existing candidate reports by default, exposes only `run_ref` and `manifest.display` artifacts through the API, and sanitizes private raw/interim/processed/report paths from candidate summaries and failed logs.

### Docs and Diff Hygiene

- `README.md`, `ARCHITECTURE.md`, `EVALUATION_INTERFACE.md`, `RESEARCH_CARD.md`, `PROJECT_STATUS.md`, `docs/architecture/`, and `docs/code-map/` describe the same current contract and phase boundary.
- Search changed files for retired terms and fields before final response: `artifacts.xlsx`, old `total_*_portfolio_return` fields, old schema versions, retired endpoint names, and old UI terminology.
- Review untracked files explicitly. Project-local skills and new source files must be included; generated runs, logs, screenshots, caches, build output, and temporary scripts must not be included.
- Root handoff rows may be closed only after the root workspace has actually synchronized the signal.

## Testing and Browser Acceptance Matrix

Use automated tests to prove contracts and state logic. Use browser acceptance to prove the real local app can be used without layout, download, or stale-state regressions.

### Automated Contract Tests

For artifact and API changes, cover these surfaces:

- producer and loader: schema version, required tables, required columns, `manifest.display` validation, path safety, and rejection of retired fields;
- API and download: run list, run evidence, table preview, direct CSV download, selected ZIP download, all-items ZIP download, comparison download, unknown items, unsafe paths, and cache behavior;
- uploaded-run deletion: one positive uploaded-panel cleanup path, plus negative tests for private/generated runs, missing manifests, unknown roots, unsafe run IDs, and out-of-root traversal attempts;
- frontend state units: run identity matching, first valid combination, combination filtering, disabled tab correction, display-driven table specs, chart range readouts, chart empty states, and selected artifact URLs;
- upload or create-run flow: required columns, `pnl_return_column`, engine selection, time breakdowns, and user-facing validation messages.

For every new validation rule, add one positive path and at least one negative path that proves stale or unsupported input is rejected. For every retired behavior, add an absence or rejection test when feasible; keyword searches are only a backstop.

### Browser Acceptance

For frontend evidence-workbench changes, run a real-browser pass against local FastAPI and Vite services:

- default result browser page loads and renders the selected run;
- run switching updates title, summary cards, interpretation, charts, metric details, CSV preview, report mode, comparison state, and download entries;
- combination switching updates every combination-dependent surface, not just the primary chart;
- full report mode expands the expected evidence surfaces without internal implementation notes;
- CSV preview defaults collapsed, expands on demand, and shows no more than the intended preview row limit;
- all-items download starts a browser ZIP download without `Failed to fetch`;
- Factor Studio shows A-share enabled and crypto planned/disabled;
- Pipeline Builder validates required fields and does not imply deployment or trading;
- Pipeline Builder is currently a snapshot flow, not a full A-share pipeline executor. Browser copy, docs, and tests should not claim one-click full execution until build/evaluate orchestration is wired.
- at least one narrow viewport is checked for overlap, clipped labels, and unusable controls.

After browser acceptance, stop the local services and list any validation temporary outputs such as runtime logs, Playwright downloads/screenshots, and download cache files before cleanup. Follow project deletion rules: remove only exact approved paths or single explicitly named files. Do not remove generated runs, profiles, or dependencies as part of validation cleanup.

## Generated Runs

`reports/private_real/`, `reports/runs/`, and `reports/profiles/` are generated evidence, not authority docs. If old runs cannot load after a schema change, explain the regeneration need instead of weakening the loader.

For large real or private runs, summarize with deterministic scripts or compact CSV/Markdown summaries. Do not ask the LLM to parse huge run directories as the primary analysis mechanism.

## Verification

Use the narrowest relevant checks first. Before claiming a broad maintenance change is ready, prefer this sequence:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
git diff --check
```

Run frontend commands from `web/`. If browser verification is needed and the bundled Playwright browser is missing, either use an installed local browser explicitly or report that visual verification was not completed.
