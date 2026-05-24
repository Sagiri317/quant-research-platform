---
name: quant-research-platform-maintenance
description: Use when changing quant-research-platform artifact schemas, manifest.display, evidence API, download packaging, React evidence UI, A-share pipeline runs, Factor Factory runs, generated runs, or docs that describe those contracts.
---

# Quant Research Platform Maintenance

## Purpose

Keep generated artifacts as the source of truth, `manifest.display` as the API/UI/download contract, and the React workbench as a consumer of those contracts. Do not restore hidden frontend business logic, schema fallbacks, broad filesystem cleanup, or private-data exposure.

Project `AGENTS.md` owns hard safety boundaries: local-only scope, approved Conda environment, data/API/secrets limits, forbidden operations, private CSMAR boundaries, and editable areas. This skill owns the operational workflow for artifact/API/frontend/A-share/Factor Factory contract changes and the docs that describe them.

## Start Here

1. Read project `AGENTS.md`.
2. Inspect only the requested surface and its linked contract files.
3. If the change touches schema, API payloads, downloads, frontend evidence state, A-share runs, Factor Factory runs, or generated evidence, read [references/contracts-and-verification.md](references/contracts-and-verification.md).
4. Keep `PROJECT_STATUS.md` and workspace-root handoff rules as defined by `AGENTS.md`; do not edit root workspace files unless explicitly asked.

Use PowerShell `Select-String` if `rg` is unavailable or denied.

## Current Contract Map

| Surface | Start here |
| --- | --- |
| Evaluation artifact schema `0.5` | `src/quant_research_platform/artifacts/artifact_schema.py`, `artifact_writer.py`, `artifact_loader.py`, `manifest_validation.py` |
| `manifest.display` | `src/quant_research_platform/artifacts/display_manifest.py`, `manifest_validation.py` |
| Artifact ZIP packaging | `src/quant_research_platform/artifacts/artifact_package.py`, `tests/artifacts/test_artifact_package.py` |
| Evidence API and downloads | `src/quant_research_platform/interfaces/api/routes/`, `interfaces/api/services/`, `interfaces/api/app.py`, `interfaces/api/run_index.py` |
| A-share runtime paths | `src/quant_research_platform/markets/implementations/a_share/runtime/storage.py`, `runtime/check_processed_space.py` |
| A-share private evaluation | `src/quant_research_platform/markets/implementations/a_share/evaluation/run_factor_evaluation.py`, `evaluation/evaluation_adapter.py` |
| A-share Pipeline Builder API | `src/quant_research_platform/interfaces/api/routes/pipeline.py`, `interfaces/api/services/pipeline_service.py`, `src/quant_research_platform/pipelines/a_share/` |
| Factor Factory schema `0.1` | `src/quant_research_platform/research/factor_factory/`, `src/quant_research_platform/interfaces/api/routes/factor_factory.py` |
| Frontend evidence workbench | `web/src/App.tsx`, `ResultsView.tsx`, `RunCharts.tsx`, `ArtifactDownloads.tsx`, `CsvBrowser.tsx`, `runEvidenceState.ts` |
| Frontend creation surfaces | `web/src/CreateRunPanel.tsx`, `CreateRunGuide.tsx`, `FactorStudio.tsx`, `FactorFactoryRuns.tsx`, `PipelineBuilder.tsx`, `api.ts` |
| Local scripts | `scripts/check_all.ps1`, `scripts/start_private_workbench.ps1`, `scripts/start_api_private_runs.ps1`, `scripts/start_web.ps1`, `scripts/stop_workbench.ps1`, `scripts/private_run_roots.ps1`, `scripts/phase1_acceptance.py` |
| Docs and maps | `README.md`, `ARCHITECTURE.md`, `EVALUATION_INTERFACE.md`, `RESEARCH_CARD.md`, `PROJECT_STATUS.md`, `docs/architecture/`, `docs/code-map/` |

## Core Rules

- Evaluation runs are `artifact_type="evaluation_set"` with `artifact_schema_version="0.5"`.
- Factor Factory runs are `artifact_type="factor_factory_run"` with `artifact_schema_version="0.1"`.
- Do not conflate the two schema versions; tests, API validation, docs, and frontend types must name the artifact type they are checking.
- `manifest.display` remains the only UI/download exposure contract for run artifacts.
- Artifact ZIP assembly belongs in `artifacts/artifact_package.py`; API download services should orchestrate responses instead of owning artifact packaging rules.
- Preserve API-visible evaluation run identity as `(source_alias, run_id)` across run lists, evidence routes, CSV preview, direct downloads, ZIP cache paths, comparison payloads, and frontend route state.
- Pipeline Builder defaults to snapshot creation through `POST /api/pipeline-snapshots`. Executable A-share pipeline requests use `POST /api/pipeline-runs` with `start_from`; every executable run continues through evaluation.
- Factor Studio and Pipeline Builder must keep A-share enabled and crypto/other markets disabled or planned until real modules exist.
- Generated runs under `reports/private_real/`, `reports/runs/`, and `reports/profiles/` are evidence, not authority docs. Regenerate stale runs instead of weakening loaders.

## Workflow

1. Define the contract first: artifact type, schema version, table names, column names, `combination_id` semantics, `manifest.display` tables/downloads/presets/charts, and API-visible identity.
2. Update producer, loader validation, lightweight run index, API payload/download behavior, frontend consumers, tests, docs, and code maps in the same round when the contract changes.
3. Remove retired compatibility in one pass when old generated runs are no longer supported. Search for stale schema versions, retired summary fields, old table names, and frontend-only shortcuts.
4. Review the diff by surface: artifact contract, API/downloads, frontend UI, A-share pipeline/runtime, Factor Factory, tests/fixtures, docs/skills, generated outputs, and gitignore.
5. For docs-only drift, verify against current code paths before preserving command examples or path names.

## Verification

Run the narrowest relevant checks first. For broad contract work, prefer:

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

Use `scripts/phase1_acceptance.py` only for Phase 1 private acceptance tasks that intentionally touch `reports/private_real/`: `rename-baseline`, `run`, `compare`, or `factory-smoke`. It is not a routine unit-test substitute, and it should not be used for public-safe or source-only documentation checks.

For frontend evidence-workbench changes, run browser acceptance only when the UI behavior changed. Start/stop local services with the flat `scripts/` PowerShell launchers, then list any logs, downloads, screenshots, or cache outputs before cleanup. Follow project deletion rules exactly.

## Reference Loading

Read [references/contracts-and-verification.md](references/contracts-and-verification.md) when you need detailed exit criteria for:

- evaluation-set schema `0.5`;
- Factor Factory schema `0.1`;
- API/download validation;
- frontend state and browser acceptance;
- A-share pipeline/runtime boundaries;
- Phase 1 acceptance script usage;
- skill-governance and global-skill sedimentation recommendations.
