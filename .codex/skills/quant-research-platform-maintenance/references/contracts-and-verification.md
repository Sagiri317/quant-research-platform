# Quant Research Platform Contracts and Verification

Load this reference when changing or reviewing artifact/API/frontend/A-share/Factor Factory contracts. Keep one-time run results, benchmark numbers, debug logs, and deleted temporary artifacts out of this file.

## Artifact Types

### `evaluation_set` schema `0.6`

- Source files: `src/quant_research_platform/artifacts/artifact_schema.py`, `artifact_writer.py`, `artifact_loader.py`, `manifest_validation.py`, `display_manifest.py`.
- Writer emits `artifact_schema_version="0.6"` and `artifact_type="evaluation_set"` in `manifest.json` and `summary.json`.
- Loader rejects unsupported artifact schema versions and unsupported artifact types.
- `manifest.display.schema_version` must match the current evaluation artifact schema and must be validated at API/UI entry points.
- All display tables must map to real artifact CSV paths; all download items must be backed by `manifest.display` or an explicitly allowed metadata item.
- Required table additions intentionally invalidate stale generated runs; reject and regenerate them instead of adding optional loader fallbacks.
- Never reintroduce `artifacts.xlsx`, old `total_*_portfolio_return` summary fields, direct frontend assumptions about undeclared files, or fallback loaders that silently accept retired schemas.

### `factor_factory_run` schema `0.1`

- Source files: `src/quant_research_platform/research/factor_factory/reports.py`, `batch_runner.py`, `src/quant_research_platform/interfaces/api/routes/factor_factory.py`, `web/src/features/factor-factory/FactorFactoryRuns.tsx`, `web/src/shared/api/api.ts`.
- Writer emits `artifact_schema_version="0.1"` and `artifact_type="factor_factory_run"`.
- API detail routes must reject wrong artifact types, wrong schema versions, missing/invalid `display`, unsafe paths, and stale manifests.
- API and frontend should expose candidate summaries through `manifest.display`; successful N7-backed summaries use `candidate_id + combination_id` plus public metrics and `run_ref`, not raw private paths.
- Factor Factory batch runs should write distinct factory run directories and avoid overwriting candidate reports by default.
- Candidate summaries and failed logs must sanitize private raw/interim/processed/report paths.

## API and Download Exit Criteria

- Run list, run evidence, CSV preview, direct CSV download, ZIP download, and comparison endpoints consume the same display contract.
- Multi-root evaluation APIs and frontend state must not collapse runs by `run_id` alone; `source_alias` is part of the identity.
- Direct CSV links validate display schema and exposed download items.
- ZIP item selection validates against the manifest before creating cache paths or writing temporary ZIP files.
- Uploaded-run deletion is limited to `summary.data_stage == "uploaded_panel"`, configured run roots, immediate child run directories, and directories containing `manifest.json`.
- Bad or stale runs are rejected or excluded at the earliest reasonable entry point.

## N8 Contract Consumer Convergence

Use this section when the work is explicitly about simplifying or converging contract consumers rather than adding a new research diagnostic.

- Shared `manifest.display` preview/download helpers may serve both evaluation runs and Factor Factory, but they must keep artifact type and schema validation explicit: evaluation-set schema `0.6` is not Factor Factory schema `0.1`.
- Factor-template discovery should be separable from pipeline snapshot/run submission. If `GET /api/factor-templates` moves out of pipeline routes, keep Factor Studio behavior stable and keep Pipeline API focused on snapshot/run.
- Frontend table preview should consume backend-provided endpoint metadata when available. Avoid duplicating backend route rules in frontend path inference.
- A-share build/evaluate service extraction should make CLI, pipeline runner, and Factor Factory thinner callers of the same service layer; it must not change `start_from` semantics or imply build-only pipeline runs.
- CSV upload is legacy/dev compatibility and not the A-share private main chain. Any renaming or hiding must preserve local debugging boundaries and deletion safety.
- Do not combine N8 cleanup with N5/N6 research-contract changes unless a written plan states the exact shared files, test boundaries, and rollback path.

## Frontend Exit Criteria

- Run switching clears stale evidence and updates title, summary cards, interpretation, charts, metric details, CSV preview, report mode, comparison state, and downloads.
- Combination switching updates every combination-dependent surface.
- Display-manifest table paths cover dimensions, metrics, portfolio, diagnostics, and breakdown tables; dimension tables such as `dimensions/combinations.csv` must have a preview route.
- Disabled or unavailable tabs reset to the first valid tab.
- Cumulative returns and IC / RankIC remain primary evidence charts. Turnover, single-period return, and cost diagnostics stay subordinate or switchable.
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

All local launch and verification helpers now live directly under `scripts/`:

- `scripts/check_all.ps1`
- `scripts/start_private_workbench.ps1`
- `scripts/start_api_private_runs.ps1`
- `scripts/start_web.ps1`
- `scripts/stop_workbench.ps1`
- `scripts/private_run_roots.ps1`
- `scripts/playwright_cli_project.ps1`
- `scripts/phase1_acceptance.py`
- `scripts/verify_code_map_flow.ps1`

Do not document stale `scripts/local/` or `scripts/build/` command paths.

`scripts/phase1_acceptance.py` is a private Phase 1 acceptance helper. Use it only when the task intentionally covers acceptance baseline renaming, 2024 acceptance run regeneration, compact comparison, or Factor Factory smoke under `reports/private_real/`. It is not a routine check for docs-only, source-only, or public-safe work.

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
- P2: user-visible inconsistency, missing required display/download coverage, incomplete negative validation, maintainability issue that is likely to cause the next N7/Factor Factory change to break.
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
