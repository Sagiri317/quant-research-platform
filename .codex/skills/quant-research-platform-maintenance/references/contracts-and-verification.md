# Quant Research Platform Contracts and Verification

Load this reference when changing or reviewing artifact/API/frontend/A-share/Factor Factory contracts. Keep one-time run results, benchmark numbers, debug logs, and deleted temporary artifacts out of this file.

## Stability Model

This reference is the project-specific and change-prone layer for the maintenance skill. Update it when schema versions, source paths, script names, API route names, browser acceptance inputs, or current verification commands change. Keep the parent `SKILL.md` focused on stable workflow principles and review behavior.

Long-term standards remain separate authority sources:

- `docs/architecture/target-architecture-and-capabilities.md` defines the target architecture and current/local-vs-future capability boundary.
- `docs/architecture/engineering-standards.md` defines durable engineering rules for module boundaries, contracts, versioning, errors, tests, generated outputs, API/frontend behavior, security, and scale.
- `docs/architecture/quant-research-standards.md` defines durable quant research rules for point-in-time data, eligibility/tradability, bias controls, factor provenance, costs, capacity, benchmarks, risk, lineage, and trading-boundary semantics.

Use this reference for current QRP contract facts; use the architecture standards for long-lived principles. Entries marked "planned" are target contracts from the schema `0.7` implementation plan and must not be treated as current-file failures before their owning slice lands.

Artifact-relative diagnostics paths such as `diagnostics/diagnostic_summary.csv` do not imply a top-level `src/quant_research_platform/diagnostics/` package. Schema `0.7` diagnostics implementation code lives under `src/quant_research_platform/evaluations/diagnostics/`.

## Current and Planned Source Map

| Surface | Start here |
| --- | --- |
| Evaluation artifact schema `0.7` planned until Slice 2 | `src/quant_research_platform/artifacts/schema_registry.py`, `artifact_schema.py`, `artifact_writer.py`, `artifact_loader.py`, `manifest_validation.py` |
| Evaluation diagnostics planned until schema `0.7` slices land | `src/quant_research_platform/evaluations/diagnostics/` |
| `manifest.display` | `src/quant_research_platform/artifacts/display_manifest.py`, `manifest_validation.py` |
| Artifact ZIP packaging | `src/quant_research_platform/artifacts/artifact_package.py`, `tests/artifacts/test_artifact_package.py` |
| Evidence API and downloads | `src/quant_research_platform/interfaces/api/routes/`, `interfaces/api/services/`, `interfaces/api/app.py`, `interfaces/api/run_index.py` |
| A-share runtime paths | `src/quant_research_platform/markets/implementations/a_share/runtime/storage.py`, `runtime/check_processed_space.py` |
| A-share private evaluation | `src/quant_research_platform/markets/implementations/a_share/evaluation/run_factor_evaluation.py`, `evaluation/evaluation_adapter.py` |
| A-share Pipeline Builder API | `src/quant_research_platform/interfaces/api/routes/pipeline.py`, `interfaces/api/services/pipeline_service.py`, `src/quant_research_platform/pipelines/a_share/` |
| Factor Factory schema `0.1` | `src/quant_research_platform/research/factor_factory/`, `src/quant_research_platform/interfaces/api/routes/factor_factory.py` |
| Frontend evidence workbench | `web/src/features/evidence/`, `web/src/shared/api/api.ts`, `web/src/shared/downloads.ts` |
| Frontend creation surfaces | `web/src/features/uploads/`, `web/src/features/factor-studio/`, `web/src/features/factor-factory/`, `web/src/features/pipeline/`, `web/src/features/runs/` |
| Local scripts | see [Local Scripts](#local-scripts) |
| Docs and maps | `README.md`, `ARCHITECTURE.md`, `EVALUATION_INTERFACE.md`, `RESEARCH_CARD.md`, `PROJECT_STATUS.md`, `docs/architecture/`, `docs/code-map/` |

## Artifact Types

### `evaluation_set` schema `0.7`

- Planned/current source files after Slice 2: `src/quant_research_platform/artifacts/schema_registry.py`, `artifact_schema.py`, `artifact_writer.py`, `artifact_loader.py`, `manifest_validation.py`, `display_manifest.py`.
- Writer emits `artifact_schema_version="0.7"` and `artifact_type="evaluation_set"` in `manifest.json` and `summary.json`.
- Loader rejects unsupported artifact schema versions and unsupported artifact types.
- `manifest.display.schema_version` must match the current evaluation artifact schema and must be validated at API/UI entry points.
- All display tables must map to real artifact CSV paths; all download items must be backed by `manifest.display` or an explicitly allowed metadata item.
- Schema `0.7` direct and ZIP downloads use display refs `{kind, key}`; raw artifact-relative strings are not the public item contract.
- Schema `0.7` portable metadata uses `run_context.json`; `config.json` and `lineage.json` are retired names used only for negative/static coverage.
- Schema `0.7` primary cumulative-return charts render API-provided derived chart payloads from registry `ChartSpec`; do not restore `portfolio/cumulative_returns.csv`.
- Required table additions intentionally invalidate stale generated runs; reject and regenerate them instead of adding optional loader fallbacks.
- Never reintroduce `artifacts.xlsx`, old `total_*_portfolio_return` summary fields, direct frontend assumptions about undeclared files, or fallback loaders that silently accept retired schemas.

### `factor_factory_run` schema `0.1`

- Source files: `src/quant_research_platform/research/factor_factory/reports.py`, `batch_runner.py`, `src/quant_research_platform/interfaces/api/routes/factor_factory.py`, `web/src/features/factor-factory/FactorFactoryRuns.tsx`, `web/src/shared/api/api.ts`.
- Writer emits `artifact_schema_version="0.1"` and `artifact_type="factor_factory_run"`.
- API detail routes must reject wrong artifact types, wrong schema versions, missing/invalid `display`, unsafe paths, and stale manifests.
- API and frontend should expose candidate summaries through `manifest.display`; successful schema `0.7` evaluation-backed summaries use `candidate_id + combination_id` plus public metrics and `run_ref`, not raw private paths.
- Factor Factory batch runs should write distinct factory run directories and avoid overwriting candidate reports by default.
- Candidate summaries and failed logs must sanitize private raw/interim/processed/report paths.

## API and Download Exit Criteria

- Run list, run evidence, CSV preview, direct CSV download, ZIP download, and comparison endpoints consume the same display contract.
- Multi-root evaluation APIs and frontend state must not collapse runs by `run_id` alone; `source_alias` is part of the identity.
- Direct CSV links validate display schema and exposed download items.
- Display tables that declare `supports_combination_filter=true` must actually contain a `combination_id` column when a preview request includes `combination_id`; reject missing filter columns instead of returning an unfiltered preview.
- ZIP item selection validates against the manifest before creating cache paths or writing temporary ZIP files.
- Uploaded-run deletion is limited to `summary.data_stage == "uploaded_panel"`, configured run roots, immediate child run directories, and directories containing `manifest.json`.
- Bad or stale runs are rejected or excluded at the earliest reasonable entry point.

## N8 Contract Consumer Convergence

Use this section when reviewing, extending, or protecting the already-merged N8 contract-consumer surfaces rather than adding a new research diagnostic.

- Shared `manifest.display` preview/download helpers may serve both evaluation runs and Factor Factory, but they must keep artifact type and schema validation explicit: evaluation-set schema `0.7` is not Factor Factory schema `0.1`.
- Factor Factory active evaluation-set consumers must migrate to schema `0.7`; `factor_factory_run` remains schema `0.1`, and `factor_summary.csv` remains the downstream comparison contract.
- Factor-template discovery should be separable from pipeline snapshot/run submission. If `GET /api/factor-templates` moves out of pipeline routes, keep Factor Studio behavior stable and keep Pipeline API focused on snapshot/run.
- Frontend table preview should consume backend-provided endpoint metadata when available. Avoid duplicating backend route rules in frontend path inference.
- A-share build/evaluate service extraction should make CLI, pipeline runner, and Factor Factory thinner callers of the same service layer; it must not change `start_from` semantics or imply build-only pipeline runs.
- CSV upload is legacy/dev compatibility and not the A-share private main chain. Any renaming or hiding must preserve local debugging boundaries and deletion safety.
- Do not re-open or replace N8 cleanup inside N5/N6 research-contract changes unless a written plan states the exact shared files, test boundaries, and rollback path.

## Frontend Exit Criteria

- Run switching clears stale evidence and updates title, summary cards, interpretation, charts, metric details, CSV preview, report mode, comparison state, and downloads.
- Combination switching updates every combination-dependent surface.
- Display-manifest table paths cover dimensions, metrics, portfolio, diagnostics, and breakdown tables; dimension tables such as `dimensions/combinations.csv` must have a preview route.
- Disabled or unavailable tabs reset to the first valid tab.
- API-derived cumulative returns and IC / RankIC remain primary evidence charts. Turnover, single-period return, and cost diagnostics stay subordinate or switchable.
- `web/src/features/factor-studio/FactorStudio.tsx` and `web/src/features/pipeline/PipelineBuilder.tsx` must keep A-share as the enabled market and crypto/other markets disabled or planned until real modules exist.
- Frontend dependencies must follow `shell -> features -> shared`; `shared/*` must not import from `features/*` or `shell/*`.
- Frontend creation surfaces should not imply deployment, trading, paper trading, or one-click full raw-data rebuilds.

## A-share Pipeline and Runtime Boundaries

- A-share processed data location helpers live under `src/quant_research_platform/markets/implementations/a_share/runtime/`.
- A-share private evaluation should use `python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation`.
- Use `--start-year/--end-year` to locate processed factor files and `--start-date/--end-date` for `trade_date` row filtering.
- Generic `evaluate-parquet` is only for evaluation-layer, adapter, or non-A-share local Parquet checks. It is not proof that the A-share adapter selected the right coverage file, date-filtered correctly, wrote label availability summaries, or preserved private run metadata.
- `POST /api/pipeline-snapshots` writes configuration snapshots only: `run_config.yaml`, `warnings.json`, and `summary.md`.
- `POST /api/pipeline-runs` executes the A-share pipeline with `start_from`. Supported start layers are `existing_factors`, `rebuild_factors`, `rebuild_panel`, and `rebuild_interim`; every executable run continues through evaluation.
- The canonical executable runner is `src/quant_research_platform/pipelines/a_share/runner.py::run_a_share_pipeline`.
- Pipeline checkpoint API payloads must not expose private or local filesystem paths when replaying persisted `checkpoints.json`. Sanitize or reject unsafe text in `artifact`, `message`, `failure_code`, `key`, `label`, `status`, `warnings`, and `blocking_issues`, including private layer markers, drive paths, absolute paths, `path=C:/...` / `see(E:/...)` forms, and `..` traversal.
- Factor Factory failure reports must sanitize `failure_reason` in both `candidate_summary.csv` and `failed_candidates.json`. Do not rely only on private layer markers; also strip local absolute paths such as `C:/Users/...` or `path=C:/...`.

## Local Scripts

All local launch and verification helpers live directly under `scripts/`. Existing helpers are:

- `scripts/check_all.ps1`
- `scripts/start_private_workbench.ps1`
- `scripts/start_api_private_runs.ps1`
- `scripts/start_web.ps1`
- `scripts/stop_workbench.ps1`
- `scripts/private_run_roots.ps1`
- `scripts/playwright_cli_project.ps1`
- `scripts/phase1_acceptance.py`
- `scripts/verify_code_map_flow.ps1`

Schema `0.7` plan-owned helpers are planned until their slices implement them:

- `scripts/check_schema_07_project_acceptance.ps1`
- `scripts/check_schema_07_browser_acceptance.ps1`
- `scripts/check_schema_07_data_scale.py`
- `scripts/check_schema_07_stale_contract.py`
- `scripts/check_schema_07_docs_links.py`

Do not document stale `scripts/local/` or `scripts/build/` command paths.

After implemented, `scripts/check_schema_07_project_acceptance.ps1` is the schema `0.7` Slice 11B project-internal acceptance wrapper. It should verify API/Vite service absence with recorded PIDs plus configured/default/known alternate ports, workbench lockfiles, and command-line process records; fail closed if service ownership is uncertain; use the mandatory current generated-run scale baseline; call the A-share private evaluation entry point; run Factor Factory/factor-summary smoke when applicable; validate manifest/display/download/private-path leak behavior; and print only project-relative path/count output.

After implemented, `scripts/check_schema_07_data_scale.py` is the mandatory schema `0.7` data-scale preflight and comparison helper. It should inspect only project-local run roots, output aggregate counts/sizes/project-relative paths, and fail closed when rows, combinations, date span, or output footprint exceed the current generated-run baseline.

After implemented, `scripts/check_schema_07_browser_acceptance.ps1` is the schema `0.7` browser acceptance entry point. It uses project-local generated Factor Factory runs and ordinary evaluation runs, records local service evidence, and may orchestrate `scripts/start_api_private_runs.ps1` plus `scripts/start_web.ps1`.

`scripts/phase1_acceptance.py` is a legacy/private Phase 1 acceptance helper until migrated by the schema `0.7` plan. Do not use it as the schema `0.7` completion gate.

Use `scripts/playwright_cli_project.ps1` for Playwright CLI browser checks in this repository. It runs the global Codex Playwright wrapper from `.playwright-cli/`, keeping generated screenshots, traces, PDFs, videos, and binary response bodies under the ignored project-local Playwright workspace instead of creating top-level `output/`.

Use `scripts/verify_code_map_flow.ps1` for static checks on `docs/code-map/platform-flow.md` and `docs/code-map/platform-flow-interactive.html`: UTF-8 readability, critical Markdown terms, parsed HTML node/edge endpoints, directed-cycle detection, static node overlap detection, Bezier-only edge model, and `git diff --check`.

## Docs and Code-Map Drift Checks

When docs describe artifact, API, download, frontend, A-share pipeline, or Factor Factory contracts, verify them against code before preserving names or commands:

- Check FastAPI route coverage against `docs/code-map/platform-flow.md`; include route additions such as upload templates, delete routes, comparison, pipeline, and Factor Factory endpoints.
- Check referenced source/doc/script paths in root docs, `docs/architecture/`, and `docs/code-map/`; future-target paths must be explicitly described as planned or suggested, not current files.
- Search for retired document names, old code-map fragments, stale module paths, stale script directories, retired schema fields, and hidden frontend/backend shortcuts.
- Run Markdown link checks for root docs plus `docs/`, and use `git diff --check` before claiming the docs are synchronized.
- For docs-only work, prefer lightweight import/route checks over private acceptance runs; do not use `scripts/phase1_acceptance.py` unless private Phase 1 evidence regeneration is in scope.

## Invariant Mapping

Before implementing or reviewing a contract change, write down the enforceable invariants and map each one to the cheapest durable check:

- pytest for Python contracts: artifact schema, required files and columns, loader rejection, lightweight index rejection, route payloads, download gating, comparison behavior, and pipeline snapshot/run validation;
- Vitest for frontend contracts: selected run identity, selected combination behavior, table/chart preview fallback, disabled-tab correction, display-driven download URLs, and user-visible validation states;
- static searches for retired public contract names, private/local-only inputs leaking into API/frontend payloads, absolute private paths in reusable docs, and accidental generated-output tracking;
- browser acceptance only for user workflows that unit tests cannot prove, such as actual run switching, download initiation, and viewport usability.

If a reviewer keeps finding the same class of issue, add or tighten the durable check before doing another broad manual pass. Do not create a separate gate script unless the rule is cross-cutting, repeated, and awkward to express in pytest or Vitest.

## Review Severity and Stop Rules

Use these severities during branch review:

- P0: data loss, private-data exposure, destructive filesystem behavior, secret exposure, or a blocker that prevents the requested workflow from running.
- P1: wrong results, wrong run identity, wrong artifact/API/frontend contract, low-memory path that can still full-load large artifacts, or stale validation that makes invalid runs API-visible.
- P2: user-visible inconsistency, missing required display/download coverage, incomplete negative validation, maintainability issue that is likely to cause the next schema/Factor Factory change to break.
- P3: naming polish, historical doc cleanup, optional hardening, or future-proofing that does not affect the current contract.

A confirmed issue is one that is reproducible from code, tests, generated artifacts, or a specific documented contract. Fix confirmed P0/P1/P2 issues in the current round when they are inside the requested surface. For P3 findings, either fix them if the change is small and isolated or record them as residual risk; do not keep reopening the whole branch for low-impact polish.

After a fix, do a focused re-review of the touched surface and rerun the narrowest relevant check. A full cold-start re-review is needed when the fix changes shared contracts, loader/API/frontend boundaries, data-generation behavior, or the previous review missed a whole class of issues.

Stop the review loop only when:

- every enforceable invariant has pytest, Vitest, or a named static check;
- narrow checks for the touched surface pass;
- the relevant broad verification profile passes;
- one final cold-start review finds no confirmed P0/P1/P2 issue;
- `git status` and `git diff --check` show no accidental generated/private output problem.

## Automated Test Surfaces

For every new validation rule, add one positive path and at least one negative path when code changes are in scope. For retired behavior, prefer an absence or rejection test; keyword searches are only a backstop.

Useful surfaces:

- producer and loader: schema version, artifact type, required tables, required columns, `manifest.display` validation, path safety, and rejection of retired fields;
- API and downloads: run list, run evidence, table preview, direct CSV download, selected ZIP download, all-items ZIP download, comparison download, unknown items, unsafe paths, and cache behavior;
- uploaded-run deletion: positive uploaded-panel cleanup plus negative generated/private/missing-manifest/unknown-root/unsafe-run/out-of-root cases;
- frontend state: run identity matching, first valid combination, combination filtering, disabled tab correction, display-driven table specs, chart range readouts, chart empty states, and selected artifact URLs;
- create-run and pipeline flow: required columns, `return_source_column`, kernel selection, `start_from`, retired-field rejection, time breakdowns, and user-facing validation messages;
- Factor Factory: schema `0.1`, `factor_factory_run` API validation, distinct run directories, sanitized reports, and frontend listing/detail behavior.

## Browser Acceptance

Use browser acceptance when frontend behavior changes:

- API-backed workbench Playwright acceptance must start both the local FastAPI API service and the Vite frontend service before testing; do not treat an API-offline `Failed to fetch` state as acceptance evidence unless the task is explicitly UI-only.
- After implemented, schema `0.7` browser acceptance must use `scripts/check_schema_07_browser_acceptance.ps1` against project-local generated Factor Factory and ordinary evaluation runs.
- default result browser page loads and renders the selected run;
- run and combination switching update all dependent surfaces;
- full report mode expands expected evidence surfaces without internal implementation notes;
- CSV preview defaults collapsed, expands on demand, and respects preview row limits;
- all-items download starts a ZIP download without `Failed to fetch`;
- Factor Studio shows A-share enabled and crypto planned/disabled;
- Pipeline Builder validates required fields, defaults to snapshot, and does not imply deployment or trading;
- Executed pipeline runs refresh `/api/runs`, select `pipeline/<run_ref or run_id>`, switch to the evidence workbench, and show a non-mojibake success message.
- at least one narrow viewport is checked for overlap, clipped labels, and unusable controls.

After browser acceptance, stop local services and list validation outputs before cleanup. Remove only exact approved paths or single explicitly named files, and do not remove generated runs, profiles, dependencies, or private outputs as incidental cleanup.

## Skill and Memory Sedimentation

- Keep global skills for cross-project rules: Windows/Conda command forms, memory hygiene, generic skill authoring, and repository-agnostic review patterns.
- Keep hard private-data boundaries in project `AGENTS.md`. Keep this project skill/reference for `quant-research-platform` contract review surfaces, local paths, artifact schema versions, and task-specific private-data acceptance checks.
- When a drift fix teaches a reusable cross-project lesson, record the recommendation in this project skill or reference first if the user did not ask for global memory. Promote it to global memory or a global skill only with explicit approval.
- Do not store transient run evidence, deleted paths, commit hashes, or temporary debug output as skill guidance.
- Prefer a lean `SKILL.md` plus linked references over a single large instruction file; add references only when they change future behavior.
