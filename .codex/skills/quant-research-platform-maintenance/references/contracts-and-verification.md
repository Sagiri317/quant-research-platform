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
- Portable artifact metadata must remain source-root independent: do not write `source_alias`, runtime URLs, host names, local API route aliases, or cache paths into `manifest.json`, `summary.json`, `run_context.json`, or display descriptors unless the current schema explicitly allows it. Add runtime-only fields such as `preview_url` in the API payload.
- Direct CSV links validate display schema and exposed download items.
- Display tables that declare `supports_combination_filter=true` must actually contain a `combination_id` column when a preview request includes `combination_id`; reject missing filter columns instead of returning an unfiltered preview.
- ZIP item selection validates against the manifest before creating cache paths or writing temporary ZIP files.
- Uploaded-run deletion is limited to `summary.data_stage == "uploaded_panel"`, configured run roots, immediate child run directories, and directories containing `manifest.json`.
- Bad or stale runs are rejected or excluded at the earliest reasonable entry point.
- Chart and table previews should use API/display-declared metadata. Do not infer chart sources or direct-download paths in the frontend from truncated CSV preview rows or artifact-relative path heuristics.

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
- Before running Factor Factory, A-share factor generation, private evaluation, or pipeline execution, check whether local API/Vite workbench services are active. If ownership is unknown, fail closed and ask the user rather than auto-killing processes or running data jobs concurrently with an unowned service.
- Keep `start_from` semantics explicit: snapshot routes write configuration only, executable pipeline routes continue through evaluation, and Factor Factory should call the shared A-share service layer instead of CLI-style `main(argv)` internals.
- `POST /api/pipeline-snapshots` writes configuration snapshots only: `run_config.yaml`, `warnings.json`, and `summary.md`.
- `POST /api/pipeline-runs` executes the A-share pipeline with `start_from`. Supported start layers are `existing_factors`, `rebuild_factors`, `rebuild_panel`, and `rebuild_interim`; every executable run continues through evaluation.
- The canonical executable runner is `src/quant_research_platform/pipelines/a_share/runner.py::run_a_share_pipeline`.
- Pipeline checkpoint API payloads must not expose private or local filesystem paths when replaying persisted `checkpoints.json`. Sanitize or reject unsafe text in `artifact`, `message`, `failure_code`, `key`, `label`, `status`, `warnings`, and `blocking_issues`, including private layer markers, drive paths, absolute paths, `path=C:/...` / `see(E:/...)` forms, and `..` traversal.
- Factor Factory failure reports must sanitize `failure_reason` in both `candidate_summary.csv` and `failed_candidates.json`. Do not rely only on private layer markers; also strip local absolute paths such as `C:/Users/...` or `path=C:/...`.

## Local Scripts

Keep repository scripts as user-facing local workbench controls only:

- `scripts/start_private_workbench.ps1`
- `scripts/stop_workbench.ps1`
- `scripts/private_run_roots.ps1`

Do not add agent-only acceptance wrappers, one-off parity scripts, stale-contract search scripts, or broad verification aggregators to `scripts/` by default. Put those flows in this skill reference, in pytest/Vitest when they enforce a durable contract, or in a temporary command captured in the task summary.

Reusable agent-only helpers live under this skill's `scripts/` directory:

- `scripts/verify_code_map_flow.ps1`: code-map Markdown/HTML static verification.
- `scripts/playwright_cli_project.ps1`: Playwright CLI wrapper that keeps browser artifacts under project-local `.playwright-cli/`.

Do not document stale `scripts/local/`, `scripts/build/`, retired schema acceptance scripts, or Phase 1 sample-regeneration scripts as current entry points.

When a verification command is mainly for agents, prefer a skill script, pytest/Vitest, or a task-local command over a repository `scripts/` file. Repository scripts should be things the user plausibly runs directly, such as opening/closing the workbench or setting private run roots.

## Agent Implementation Loop

Use this loop for broad schema, artifact, API, frontend, A-share, or Factor Factory implementation goals:

1. Open or use an explicit goal when the user asks for full implementation and verification.
2. Split independent work only where file ownership is clear. Give each implementation subagent a disjoint write scope and tell it not to revert unrelated changes.
3. Keep shared contracts under the main agent or a single contract integrator: schema registry, `manifest.display`, API payload identity, generated-file indexes, and cross-surface naming.
4. Run narrow checks after each slice before broad verification.
5. After implementation, run a cold review with reviewers that did not participate in implementation. Use separate lenses for schema/artifact/API, frontend/download/workbench, and quant/private-data semantics when the surface is broad.
6. Fix confirmed P0/P1/P2 issues, rerun affected checks, close the old reviewer, and start a new focused cold review. Repeat until no confirmed P0/P1/P2 remains.
7. Treat P3 as optional cleanup unless it affects the current contract or the user explicitly asks to address it.

Do not continue broad plan/spec review once implementation is already approved unless a new P0/P1/P2 blocker appears.

## Spec And Plan Review Loop

Use this loop before implementing large specs or irreversible contracts:

1. Draft the spec or implementation plan with explicit non-goals, data/privacy boundaries, generated-output rules, and acceptance gates.
2. Review with separate lenses: schema/field consistency, engineering/API/display contract, quant semantics/private boundary, and frontend/workbench behavior where relevant.
3. Fix confirmed P0/P1/P2 findings and do one fresh cold review before moving to implementation.
4. If repeated review keeps finding the same class of issue, convert the issue into schema registry rules, pytest/Vitest coverage, or a static check instead of expanding prose.
5. Once implementation is approved, stop broad plan/spec review and execute the plan.

## Agent Verification Playbook

Prefer durable tests over scripts. When an agent needs a temporary verification method:

- Data-scale gate: inspect only aggregate counts, sizes, date spans, combination counts, and project-relative run paths. Do not print row-level private data. Compare against a known generated-run baseline before larger project-internal runs.
- Browser acceptance: verify API/Vite ports are not already owned by unknown processes; start local API and Vite with recorded PIDs; run Playwright against the workbench; stop only recorded PIDs; list generated browser artifacts before cleanup.
- Stale-contract search: search for retired schema versions, table names, routes, metadata files, raw download item strings, and compatibility wording. Classify remaining hits as current, historical, negative-test, plan/spec, removed, or current-facing stale. Do not blanket-allowlist `docs/`, `AGENTS.md`, or skill references; they may contain current instructions that need repair.
- Private-path leak check: search portable artifacts, API payload fixtures, frontend display payloads, downloads, reports, and logs for local drive paths, private layer markers, credentials, source aliases where disallowed, and row-level private data.
- Code-map check: verify Markdown/HTML code-map links and graph invariants directly or through a temporary local command; keep the detailed procedure in `references/code-map-flow.md`.
- Windows command-form check: avoid long multiline `conda run ... python -c` diagnostics. Prefer a checked helper, direct module invocation through `conda run -n analysis python -m ...`, or a short project-local script when the diagnostic needs structured output.

If a temporary verification proves a durable invariant, migrate it to pytest, Vitest, or a small library helper instead of preserving a standalone script.

## Generated Output Cleanup

Before deleting generated, private, cache, or acceptance outputs:

1. List exact project-relative paths, file counts, sizes, and last modified times.
2. Search current docs, code, tests, run roots, API config, and skill references for references to the path.
3. Classify each path as current evidence, historical evidence, cache, temporary parity output, stale generated run, or unknown.
4. Keep current evidence until the replacement run and docs are in place.
5. Cache directories such as download ZIP caches can be removed after exact-path approval because they are regenerated on demand.
6. Acceptance-only directories can be removed after exact-path approval only when no current docs, tests, wrappers, or workbench run roots reference them.
7. Stale generated runs should be regenerated under the current schema when needed; do not weaken loader, API, frontend, or display validation to keep them visible.
8. Never delete raw/interim/processed private data, credentials, dependency directories, or unknown generated roots through cleanup convenience steps.

## Docs and Code-Map Drift Checks

When docs describe artifact, API, download, frontend, A-share pipeline, or Factor Factory contracts, verify them against code before preserving names or commands:

- Check current-facing root documents together: `AGENTS.md`, `README.md`, `ARCHITECTURE.md`, `EVALUATION_INTERFACE.md`, `PROJECT_STATUS.md`, `docs/architecture/`, `docs/code-map/`, and this skill reference. Historical plans/specs can remain historical, but they must not be presented as current commands or current artifact roots.
- Check FastAPI route coverage against `docs/code-map/platform-flow.md`; include route additions such as upload templates, delete routes, comparison, pipeline, and Factor Factory endpoints.
- Check referenced source/doc/script paths in root docs, `docs/architecture/`, and `docs/code-map/`; future-target paths must be explicitly described as planned or suggested, not current files.
- Search for retired document names, old code-map fragments, stale module paths, stale script directories, retired schema fields, and hidden frontend/backend shortcuts.
- Run Markdown link checks for root docs plus `docs/`, and use `git diff --check` before claiming the docs are synchronized.
- For docs-only work, prefer lightweight import/route checks over private acceptance runs. Historical Phase 1 sample-regeneration scripts are retired and should not be used as current completion gates.
- If a cleanup or migration changes repository scripts, mirror that in user-facing docs and this skill reference in the same round so a future agent does not revive retired wrappers.

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
- Schema `0.7` browser acceptance should use the Agent Verification Playbook above against project-local generated Factor Factory and ordinary evaluation runs.
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
