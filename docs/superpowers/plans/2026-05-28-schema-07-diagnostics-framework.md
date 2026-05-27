# Schema 0.7 Diagnostics Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete schema `0.7` / N5 v1 diagnostics as the new incompatible evaluation-set artifact contract, implemented first through a minimum vertical slice and then through Phase 1B materialized diagnostics plus Slice 11B project-internal data acceptance.

**Architecture:** Schema `0.7` remains `artifact_type="evaluation_set"` and replaces schema `0.6` at runtime. A code-owned schema registry defines files, columns, grain, unique keys, conditional rules, sentinel rules, display eligibility, and tests; writer, loader, run index, API, downloads, and frontend consume that registry through `manifest.display`. Diagnostics are post-run explanatory outputs and must not mutate weights, returns, tradable universe, or base evaluation semantics.

**Tech Stack:** Python, pandas, pyarrow, FastAPI, pytest, TypeScript, React, Vitest, Vite.

---

## Source Spec

This plan implements `docs/superpowers/specs/2026-05-27-n5-diagnostics-framework-design.md` as an execution plan. Do not continue expanding that spec while executing this plan.

## Non-Negotiable Boundaries

- Runtime names use `diagnostics`, `schema 0.7`, `Diagnostic*`, and `diagnostic_*`; do not use `N5` / `n5_` in code, API payloads, test names, display keys, or frontend copy.
- Schema `0.7` is fully incompatible with schema `0.6`; do not add compatibility wrappers, migration loaders, or UI fallbacks for old generated runs.
- Remove the legacy diagnostics route from active code/API behavior: `/api/runs/{source_alias}/{run_id}/diagnostics/{metric_name}`. Diagnostics access must go through `/tables/{table_key}/preview` and display-approved downloads only. This is not permission to delete files, generated runs, or directories.
- Keep `manifest.display` as the only artifact exposure contract for API evidence, preview, direct download, ZIP download, frontend tables, frontend charts, and download presets.
- Do not expose private CSMAR paths, real-data paths, credentials, API keys, absolute machine paths, external API references, deployment controls, paper trading, live trading, broker/exchange integration, or real trading automation.
- `source_alias` remains API/runtime identity only and must not be serialized into portable artifact metadata.
- `DiagnosticsSpec` and `DiagnosticPolicy` are diagnostic-only in v1: they may explain, slice, warn, and mark unavailable, but must not change universe, signal, weights, returns, or cumulative returns.
- Conditional diagnostics views are registered early with path/key/status/display eligibility, but generation can land in later slices only after registry, writer, loader, display, API, download, frontend, and tests all cover the view.

## Deferred / Do Not Implement In This Plan

- Do not implement or register deferred diagnostics artifacts `diagnostics/stress_test_summary.csv`, `diagnostics/capacity_curve.csv`, or `diagnostics/crowding_proxy.csv` in schema `0.7` v1.
- Do not add optimizer, portfolio-policy, simulation-policy, execution simulator, vendor/broker/exchange API, API-key handling, deployment controls, paper trading, live trading, order routing, or production automation work.
- Phase 1 is split into Phase 1A and Phase 1B inside the same implementation goal: Phase 1A lands the schema/API/display contract shell; Phase 1B completes schema `0.7` / N5 v1 diagnostics depth through reviewable materialized-calculation and conditional-view slices. Deferred artifacts listed above remain out of scope.

## Contract Decisions To Lock Before Slice 1

- Schema version: `artifact_schema_version = "0.7"`.
- Artifact type: `artifact_type = "evaluation_set"`.
- Required metadata files: `manifest.json`, `summary.json`, `run_context.json`.
- Optional metadata file: `report.md`; if absent, it must not appear in `manifest.display.downloads.items` or ZIP presets.
- Removed from schema `0.7` artifact contract: separate `config.json` and `lineage.json`.
- Run identity fields: `summary.json.run_id` and `run_context.json.provenance.run_id` must exist and match. Schema `0.7` portable artifacts must not serialize `evaluation_id`; migrate active consumers to `run_id` and keep any historical `evaluation_id` references only in explicit rejection/historical coverage. If an internal code alias is unavoidable during refactor, it is non-portable and must not appear in `summary.json`, `run_context.json`, `manifest.json`, API payloads, frontend fixtures, or downloads.
- Required base tables for the minimum vertical slice:
  - `dimensions/combinations.csv`
  - `metrics/ic.csv`
  - `metrics/grouped_returns.csv`
  - `metrics/evaluation_metrics_summary.csv`
  - `portfolio/returns.csv`
  - `input_quality/coverage_summary.csv`
  - `dimensions/diagnostic_policies.csv`
  - `diagnostics/diagnostic_summary.csv`
  - `diagnostics/diagnostic_metrics.csv`
- Conditional table registry entries in the first stage:
  - `dimensions/portfolio_assumptions.csv`
  - `portfolio/weights.parquet`
  - `diagnostics/tradability_diagnostics.csv`
  - `diagnostics/tradability_summary.csv`
  - `diagnostics/scenario_metrics.csv`
  - `diagnostics/exposure_capacity_summary.csv`
  - `diagnostics/portfolio_risk_summary.csv`
  - `diagnostics/diagnostic_observations.parquet`
- Time-breakdown generated registry entries for current `time_breakdowns` support:
  - `breakdowns/year/evaluation_metrics_summary.csv`, display key `evaluation_metrics_summary_by_year`
  - `breakdowns/quarter/evaluation_metrics_summary.csv`, display key `evaluation_metrics_summary_by_quarter`
  - `breakdowns/month/evaluation_metrics_summary.csv`, display key `evaluation_metrics_summary_by_month`
  - Each table has grain `combination_id + period_type + period + metric_family + metric_name`, unique key `combination_id,period_type,period,metric_family,metric_name`, stable sort key `period_type,period_start,combination_id,metric_family,metric_name`, and required columns `combination_id`, `period_type`, `period`, `period_start`, `period_end`, `metric_family`, `metric_name`, `metric_value`, `unit`, `n_observations`, `diagnostic_status`, `status_reason`, and `warning`. Old summary-specific breakdown files and keys remain retired.
- Chart registry entries are first-class schema `0.7` contract objects, not frontend conventions. The primary cumulative chart has key `primary_cumulative_return`, kind `derived_cumulative_return`, source table key `returns`, source path `portfolio/returns.csv`, x column `timestamp`, grouping/filter column `combination_id`, input series `net_return`, transform `compound_cumulative_return`, output series `cumulative_net_return`, and supports combination filtering. It is API-derived and never materialized as `portfolio/cumulative_returns.csv`.
- Minimum diagnostics-disabled behavior: if diagnostics are absent or disabled, schema `0.7` still writes required diagnostics contract files with `disabled` / `skipped` status rows and no conditional materialized views.
- Common API validation/error envelope: all schema `0.7` writer/loader/API request validation errors exposed through FastAPI use one JSON shape, `{"error": {"error_code": string, "message": string, "field": string | null, "details": object}}`. Field paths use dotted paths with bracket indexes, for example `diagnostics.materialized_views[0]`; `details` is `{}` when empty; multi-error validation is represented as `details.errors` containing objects with the same `error_code` / `message` / `field` / `details` fields. HTTP status mapping is fixed before route work: `400` malformed JSON/query shape, `422` semantic contract validation, `404` missing run/file/route, `409` duplicate or conflicting run identity, and `500` only for unexpected server failures.
- Shared TypeScript contract must mirror the error envelope as `ApiErrorEnvelope` and all frontend API helpers/tests must consume that type rather than ad hoc error strings.

## Schema 0.6 Artifact Identity Decisions

Every active schema `0.6` artifact key/path must be classified before runtime switch. This table is the retired schema `0.6` migration decision ledger for implementation, tests, and stale-contract checks. It classifies retired identities only; current schema `0.7` field lists, grains, keys, `required_if` / `nullable_if` rules, display/download eligibility, and generated-file indexes remain owned by `schema_registry.py` and tests.

| Schema `0.6` identity | Schema `0.7` decision |
| --- | --- |
| `metrics/ic.csv` | Migrate as required base table with registry/display key `ic`. |
| `metrics/grouped_returns.csv` | Migrate as required base table with registry/display key `grouped_returns`. |
| `metrics/ic_summary.csv` / display key `ic_summary` | Retire; replace summary rows with `metrics/evaluation_metrics_summary.csv` using `metric_family` / `metric_name`. |
| `metrics/group_spread_summary.csv` / display key `group_spread_summary` | Retire; replace group-spread summary rows with `metrics/evaluation_metrics_summary.csv`. |
| `metrics/performance_summary.csv` / display key `performance_summary` | Retire; replace performance summary rows with `metrics/evaluation_metrics_summary.csv` and portfolio metrics derived from required returns. |
| `summaries/evaluation_summary.csv` / display key `evaluation_summary` | Retire as an artifact file; use `summary.json` for run-level summary and `metrics/evaluation_metrics_summary.csv` for metric rows. |
| `portfolio/returns.csv` | Migrate as required base table and the source table for return/cumulative-return chart display. |
| `portfolio/cumulative_returns.csv` / display key `cumulative_returns` | Retire as a materialized artifact file; schema `0.7` chart/display code must derive cumulative-return series from `portfolio/returns.csv` through registry/display metadata and must reject manifests that expose this old path. |
| `input_quality/label_availability.csv` / display key `label_availability` | Retire as a separate table; migrate label coverage rows into required `input_quality/coverage_summary.csv` with `input_family="label"`. |
| `input_quality/factor_coverage.csv` / display key `factor_coverage` | Retire as a separate table; migrate factor coverage rows into required `input_quality/coverage_summary.csv` with `input_family="factor"`. |
| `input_quality/tradability_coverage.csv` / display key `tradability_coverage` | Retire as a separate table; migrate tradability coverage rows into required `input_quality/coverage_summary.csv` with `input_family="tradability"`. |
| `breakdowns/{period}/{ic_summary,group_spread_summary,performance_summary}.csv` and display keys `*_summary_by_{year,quarter,month}` | Retire old summary-specific breakdown files/keys; migrate current year/quarter/month breakdown outputs into schema `0.7` registry-backed evaluation-metric breakdown table keys using `metric_family`, `metric_name`, and period columns. |
| Dynamic per-run `comparison.csv` download | Keep only as API-runtime comparison augmentation `{kind: "dynamic", key: "comparison"}` when comparison context exists. It must not be serialized into portable `manifest.json`, `manifest.display`, `manifest.artifacts`, or loader/package required-file checks. |
| `config.json` and `lineage.json` | Retire; replace portable metadata with `run_context.json`. |
| `/diagnostics/{metric_name}`, `/metrics/{metric_name}`, `/portfolio/{metric_name}`, `/breakdowns/{group}/{metric_name}` preview routes | Retire for schema `0.7`; use `/tables/{table_key}/preview` only. |

Stale-contract checks must include all identities in this table, not only `ic_summary` and `performance_summary`.

## Pre-Decided Defaults To Avoid Goal Blockers

These decisions are part of this implementation plan. Future implementation agents should not stop for user input on these points unless they need to cross an explicitly listed approval boundary.

Adopting/executing this plan is the task-specific approval for the listed read-only project-internal/private acceptance reads and the listed verification profile once, within the constraints below. It is not approval for dependency installation, deletion/move cleanup, project-external access, larger-than-baseline runs, or system/environment changes.

- Schema cutover: treat schema `0.7` incompatible replacement as approved by this plan. Schema `0.6` generated runs are rejected/excluded, no compatibility UI/API/migration fallback is added, retired preview routes are removed, `config.json` / `lineage.json` are not written, and schema `0.7` portable artifacts use `run_id` only.
- Runtime identity: API routes, API payloads, download/cache keys, and frontend route/cache state use `(source_alias, run_id)`; portable artifact metadata never serializes `source_alias` or `evaluation_id`.
- Project-internal data boundary: implementation may read project-internal data inside this repository for schema `0.7` implementation, tests, and acceptance. Private CSMAR raw/interim/processed layers are read-only inputs for this plan. Writes are limited to configured generated run/output roots, checked source/test/doc changes, and ignored local acceptance reports. Regenerating raw/interim/processed private layers, rewriting private input datasets, copying private outputs outside the project, or broad private-report mutation requires separate explicit approval. This does not allow printing row-level private data, committing private/generated outputs, reading project-external paths, using external APIs, downloading data, handling credentials/secrets, or connecting vendor/broker/exchange/paper/live trading systems.
- Diagnostics module boundary: schema `0.7` diagnostics live under `src/quant_research_platform/evaluations/diagnostics/` as the evaluation layer's post-run explanation subpackage, not as a new top-level platform domain. It may consume evaluation staging payloads and approved optional columns through `DiagnosticInputBundle`, but it must not mutate weights, returns, tradable universe, base evaluation semantics, market data layers, or order/simulation paths. It must not directly read private roots, vendor APIs, or frontend/API state; A-share market semantics stay in the A-share adapter/runtime, and generic Parquet semantics require explicit mapping.
- Optional diagnostics inputs: optional inputs such as benchmark return, industry, market cap, volatility, liquidity, and beta proxy may be used when they already exist inside project-local configured roots and are mapped through registry/input-bundle metadata. Missing optional inputs must produce unavailable/warning diagnostics rather than blocking the run. Optional inputs used for decision-facing diagnostics require timing metadata such as `as_of_time`, `effective_time`, `decision_time`, or `timing_semantics`; missing or ex-post timing marks the row unavailable for decision use (`decision_usable=false`) rather than silently treating it as point-in-time valid.
- Local API versioning decision: schema `0.7` changes the local API payloads in lockstep with backend/frontend shared Python and TypeScript types, but the local FastAPI surface remains unversioned in Phase 1. Artifact schema version plus shared API types are the version boundary. Do not introduce a separate HTTP `api_version`, `request_schema_version`, or `response_schema_version` unless a later deployable API design is approved.
- Benchmark display semantics: when benchmark inputs are absent, benchmark-relative metrics, cards, and report claims are unavailable or hidden with explicit copy; remaining diagnostics are descriptive only and must not be worded as alpha or benchmark-relative proof.
- `diagnostics/diagnostic_observations.parquet`: keep disabled by default in Phase 1.
- Factor Factory and factor-summary tooling: keep active in Phase 1 and preserve `factor_summary.csv` as the downstream comparison contract. Migrate active readers to schema `0.7`; do not retire Factor Factory, `build_factor_summary.py`, or `compare_factor_summaries.py` unless separately approved.
- Phase 1 acceptance script and synthetic demo notebook: migrate `scripts/phase1_acceptance.py` and `notebooks/00_synthetic_ashare_demo.py` to schema `0.7`; do not delete files. Any actual file deletion requires exact-path approval.
- Browser acceptance runs: use project-local generated Factor Factory runs and ordinary evaluation runs for schema `0.7` UI acceptance. Do not create a separate public synthetic browser-fixture path. Do not commit generated run artifacts unless the user approves exact tracked fixture paths.
- Project-internal acceptance: Phase 1 may run project-internal data acceptance using existing run scale or smaller. Do not exceed the current generated-run scale for rows, combinations, date span, or output footprint unless separately approved. Before Phase 1B data tests, run a read-only scale preflight that records the current generated-run maximum rows, combinations, date span, and output footprint; later project-internal tests must compare against that baseline.
- Resource scheduling: do not run frontend/backend workbench services at the same time as Factor Factory or A-share factor/evaluation pipeline runs. Before starting factor-flow or private/project-internal data runs, verify recorded API/Vite PIDs, configured/default/known alternate local API/Vite ports, project workbench lockfiles, and command-line process records are inactive. If any ownership is uncertain, fail and ask the operator to stop the exact service; do not stop unknown PIDs automatically.
- Browser services: browser acceptance may start local FastAPI and Vite services against configured project-local generated run roots, choose alternate local ports if defaults are occupied, record PIDs/ports/logs, and stop only recorded PIDs. If Playwright browsers or dependencies are missing, stop and report; do not install dependencies without approval.
- Verification budget: after narrow checks pass, implementation may run the plan's broad pytest/Ruff/frontend build/browser verification profile once. Stop before dependency installation, project-external data access, destructive cleanup, or system/environment changes.
- Retirement semantics: "retire" means migrate out of current code paths or mark historical/rejection coverage. Default to leaving generated/private runs in place and covering schema `0.6` through rejection tests, stale-contract allowlists, or explicit historical fixtures. Any actual move/cleanup, including generated runs or tracked source/docs/tests, must list exact source and destination paths and receive explicit user approval before the move; do not change `.gitignore`.
- Status/handoff: implementation may update project-local `PROJECT_STATUS.md` when the project topic/phase/boundary changes. Do not edit workspace-root planning/status files unless separately asked.
- Optional `report.md`: when absent, evidence returns `report: null` with an explicit report-unavailable state for the frontend; no report download item is emitted. Direct `/report.md` returns `404` when absent or not display-approved.
- Selected-download browser `GET` encoding: use repeated compact item refs, for example `items=table:diagnostic_summary&items=metadata_file:run_context`; URL-encode keys normally. JSON `POST` continues to use `items: [{"kind": "...", "key": "..."}]`.
- Direct download identity: schema `0.7` direct downloads use the same `RunDisplayDownloadRef={kind,key}` identity as selected ZIP downloads. Any legacy `/artifacts/{artifact_item:path}` route either resolves a display ref before touching paths or is retained only as schema `0.6` historical/rejection coverage; raw path strings are not a schema `0.7` download contract.
- `DiagnosticsOptions` for registered-but-unimplemented views: API may include them as `requestable=false, implemented=false`; Pipeline Builder hides non-requestable views by default and must not submit them.
- Uploaded-panel diagnostics UX: frontend hides diagnostics controls and omits the field. Backend accepts omitted diagnostics or `{enabled:false}` only.
- Runtime `comparison.csv`: expose only through a separate comparison download action/panel when comparison context exists. Do not include it in normal run download picker "select all" or portable manifest presets.
- Summary cards and cumulative chart: API owns summary-card aggregation and chart derivation from canonical schema `0.7` sources. Frontend renders API-provided summary cards and API-derived chart payloads. Phase 1 primary cumulative series is net cumulative return derived from `portfolio/returns.csv`; gross/cost/drawdown series are later extensions unless already present in an existing API payload.
- Docs wording: public-facing docs must describe schema `0.7` Phase 1A as the current evaluation-set artifact contract shell and Phase 1B as the complete N5 v1 post-run diagnostics implementation. Deferred artifacts remain future work.

## File Map

Create:

- `src/quant_research_platform/artifacts/schema_registry.py`: code-owned registry for schema `0.7` files, fields, grains, unique keys, sort keys, conditional rules, sentinel rules, display eligibility, and required/conditional status.
- `src/quant_research_platform/evaluations/diagnostics/__init__.py`: public evaluation diagnostics subpackage exports.
- `src/quant_research_platform/evaluations/diagnostics/spec.py`: `DiagnosticsRequest`, `DiagnosticsSpec`, normalization, disabled/options validation, and machine-readable error codes.
- `src/quant_research_platform/evaluations/diagnostics/policies.py`: `DiagnosticPolicy`, policy families, missing-input modes, allowed toggles, materialized view registry, and v1 default policy set.
- `src/quant_research_platform/evaluations/diagnostics/inputs.py`: `DiagnosticInputBundle` and staging-payload boundary helpers; this reads base evaluation payload plus approved optional panel inputs without making staging files API-visible.
- `src/quant_research_platform/evaluations/diagnostics/metrics.py`: `DiagnosticMetricSpec` support and materialized diagnostics metric calculators.
- `src/quant_research_platform/evaluations/diagnostics/runner.py`: minimal `DiagnosticRunner` that emits required disabled/summary/metrics frames and later dispatches registered policies.
- `src/quant_research_platform/evaluations/diagnostics/views.py`: conditional diagnostics view path/key/status/display eligibility registration.
- `tests/artifacts/test_schema_07_registry.py`: registry source-of-truth tests.
- `tests/artifacts/test_schema_07_artifacts.py`: writer/loader/manifest validation tests for schema `0.7`.
- `tests/evaluations/diagnostics/test_diagnostics_spec.py`: request normalization and policy validation tests.
- `web/src/features/evidence/DiagnosticTables.tsx`: generic diagnostics section backed only by `manifest.display`.
- `web/src/features/evidence/__tests__/DiagnosticTables.test.tsx`: frontend generic diagnostics rendering tests.

Modify:

- `src/quant_research_platform/artifacts/artifact_schema.py`: add schema `0.7` registry integration additively in Slice 1, then switch current evaluation-set runtime contract in Slice 2 with writer/loader/index/fixture migration.
- `src/quant_research_platform/artifacts/artifact_writer.py`: write schema `0.7` required files and tables, validate against registry, write `run_context.json`, stop writing schema `0.7` `config.json` / `lineage.json`.
- `src/quant_research_platform/artifacts/artifact_loader.py`: require schema `0.7`, `run_context.json`, run-id equality, registry column validation, required files, and display contract.
- `src/quant_research_platform/artifacts/artifact_comparison.py`: migrate comparison inputs from `config.json` / old summary tables to schema `0.7` `run_context.json` and `metrics/evaluation_metrics_summary.csv`.
- `src/quant_research_platform/artifacts/artifact_inventory.py`: migrate inventory metadata from `evaluation_id` / schema `0.6` assumptions to schema `0.7` run identity and registry tables.
- `src/quant_research_platform/artifacts/evaluation_comparison_report.py`: migrate low-memory comparison report to schema `0.7` summary and registry table names.
- `src/quant_research_platform/datasets/parquet_panel.py`: ensure generic Parquet evaluation inputs normalize absent/disabled diagnostics into the same `DiagnosticsSpec` path as A-share and uploaded-panel runs.
- `src/quant_research_platform/evaluations/evaluation_spec.py`: keep `time_breakdowns` valid for schema `0.7` and migrate its output semantics to registry/display-backed schema `0.7`.
- `src/quant_research_platform/evaluations/period_breakdowns.py`: migrate period breakdown outputs to schema `0.7` registry/display/API/download contracts; do not retire current year/quarter/month breakdowns in this plan.
- `src/quant_research_platform/artifacts/manifest_validation.py`: replace hard-coded schema `0.6` table checks with registry-driven validation.
- `src/quant_research_platform/artifacts/display_manifest.py`: build display entries from registry metadata, including generic diagnostics roles and conditional display eligibility.
- `src/quant_research_platform/artifacts/display_tables.py`: keep preview/download path helpers schema-aware and registry-backed where needed.
- `src/quant_research_platform/artifacts/artifact_type_registry.py`: expose schema `0.7` outputs/downloads from registry.
- `src/quant_research_platform/artifacts/artifact_package.py`: package display-approved schema `0.7` metadata and table items, including `run_context.json`.
- `src/quant_research_platform/artifacts/public_metadata.py`: sanitize schema `0.7` `run_context.json` and display/download metadata before writer/package/API exposure.
- `src/quant_research_platform/interfaces/api/services/context.py`: preserve comparison context while migrating `comparison.csv` to schema `0.7` API-runtime dynamic download refs outside portable manifests.
- `src/quant_research_platform/interfaces/api/run_index.py`: lightweight index must reject schema `0.6`, require schema `0.7` required files, validate headers from registry, and avoid full table loads.
- `src/quant_research_platform/interfaces/api/errors.py` and `src/quant_research_platform/interfaces/api/app.py`: define and register the app-wide local API error-envelope strategy for `HTTPException`, request validation, and project validation errors; migrate affected route tests, including Factor Factory routes if present.
- `src/quant_research_platform/interfaces/api/payloads.py`: update run payloads to schema `0.7` summary names and display-driven evidence.
- `src/quant_research_platform/interfaces/api/routes/runs.py`: remove legacy diagnostics route; keep generic table preview route and display-approved direct download behavior.
- `src/quant_research_platform/interfaces/api/routes/uploads.py` and `src/quant_research_platform/interfaces/api/services/upload_service.py`: migrate uploaded-panel run creation to schema `0.7` with diagnostics disabled by default.
- `src/quant_research_platform/interfaces/api/routes/artifacts.py`: migrate direct artifact download endpoint to schema `0.7` display-approved items, including `run_context.json` and rejection of retired metadata files.
- `src/quant_research_platform/interfaces/api/routes/downloads.py`: migrate direct and ZIP download routes to schema `0.7` display item shape and `run_context.json`.
- `src/quant_research_platform/interfaces/api/routes/comparison.py`: preserve `(source_alias, run_id)` and schema `0.7` comparison payloads.
- `src/quant_research_platform/interfaces/api/services/download_service.py`: remove direct `config.json` / `lineage.json` branches and resolve metadata/table downloads only from `manifest.display`.
- `src/quant_research_platform/interfaces/api/services/comparison_service.py`: consume schema `0.7` `run_context` / `evaluation_metrics_summary` and preserve source aliases in comparison rows.
- `src/quant_research_platform/interfaces/api/services/table_preview_service.py`: preserve bounded CSV/Parquet preview semantics and column pruning for optional Parquet detail views.
- `src/quant_research_platform/interfaces/api/services/pipeline_service.py`: add diagnostics request object to snapshot/run payloads and normalize through diagnostics module.
- `src/quant_research_platform/pipelines/a_share/config.py`: remove `diagnostics: tuple[str, ...]` semantics and store normalized diagnostics request/spec shape.
- `src/quant_research_platform/pipelines/a_share/pipeline_step_registry.py`: replace current evaluation artifact output contract entries such as `evaluation/config.json` with schema `0.7` outputs such as `evaluation/run_context.json`.
- `src/quant_research_platform/research/factor_factory/`: inspect and migrate any schema `0.6` artifact consumer, generated-run fixture, or evaluation-summary reader to schema `0.7`.
- `src/quant_research_platform/interfaces/api/routes/factor_factory.py` and related Factor Factory services if present: migrate only the Factor Factory consumers of evaluation-set artifacts to schema `0.7`; keep Factor Factory's own `factor_factory_run` artifact contract/version separate unless separately designed.
- `src/quant_research_platform/evaluations/evaluation_runner.py`: thread optional diagnostics into artifact writing without changing base evaluation semantics.
- `src/quant_research_platform/evaluations/evaluation_payload.py`: define or extend the internal non-portable staging payload that diagnostics may consume before portable artifact writing; raw frames and private input refs must not become API-visible artifacts.
- `src/quant_research_platform/markets/implementations/a_share/evaluation/evaluation_adapter.py`: pass diagnostics request/spec through A-share private evaluation path.
- `src/quant_research_platform/markets/implementations/a_share/evaluation/build_factor_summary.py`: migrate active factor-summary aggregation from schema `0.6` `config.json`, `ic_summary`, `group_spread_summary`, `performance_summary`, input-quality split files, and `evaluation_id` assumptions to schema `0.7`; preserve `factor_summary.csv` as the downstream comparison contract.
- `src/quant_research_platform/markets/implementations/a_share/evaluation/compare_factor_summaries.py`: keep `factor_summary.csv` stable for comparison and update the CLI/tests only as needed to consume migrated schema `0.7` factor-summary outputs.
- `src/quant_research_platform/interfaces/cli/commands.py`: add config/CLI diagnostics parsing only as a `DiagnosticsSpec` construction path; no separate diagnostics runner entry point.
- `web/src/shared/api/api.ts`: contract-integrator-owned shared API/type delta for diagnostics request/display types and schema `0.7` run/display types.
- `web/src/shared/testFixtures.ts`: update fixtures to schema `0.7` display tables and required diagnostics rows.
- `web/src/features/evidence/ResultsView.tsx`: render generic diagnostics sections instead of only tradability-specific UI.
- `web/src/features/evidence/useRunEvidence.ts`: fetch diagnostics tables only from display table specs; keep combination-aware refresh only for tables that opt in.
- `web/src/features/evidence/CsvBrowser.tsx`: continue generic display-driven detail preview, no path inference.
- `web/src/features/evidence/ArtifactDownloads.tsx`, `web/src/features/evidence/useArtifactDownloads.ts`, and `web/src/shared/downloads.ts` if present: migrate download item identity from raw strings to display item `{kind, key}` references.
- `web/src/shell/App.test.tsx`: migrate shell-level expectations away from old `/metrics` routes, `performance_summary` display keys, and raw artifact download paths unless explicitly classified as historical/rejection coverage.
- `web/src/features/factor-factory/`: inspect and migrate Factor Factory frontend run/evidence consumers if they read schema `0.6` summaries, downloads, or preview keys.
- `web/src/features/uploads/`: migrate uploaded-panel creation forms/hooks/API types to schema `0.7` uploaded-panel run behavior.
- `web/src/features/pipeline/PipelineBuilder.tsx`: expose minimal diagnostics options from shared types/options without frontend-owned policy JSON.
- `ARCHITECTURE.md`, `EVALUATION_INTERFACE.md`, `PROJECT_STATUS.md`, `README.md`, `AGENTS.md`, `.codex/skills/quant-research-platform-maintenance/SKILL.md`, `.codex/skills/quant-research-platform-maintenance/references/contracts-and-verification.md`, `docs/architecture/`, and `docs/code-map/`: update in the same Phase 1 merge round as producer/API/frontend contract changes, after contract tests pass but before merge.
- `REPORT.md`, `RESEARCH_CARD.md`, and notebooks/demo examples: update current-contract schema references and runnable demos to schema `0.7`; retain schema `0.6` references only as explicitly historical/rejection context.
- `scripts/check_schema_07_stale_contract.py` and `tests/scripts/test_schema_07_stale_contract.py`: add a checked stale-contract gate with explicit allowlists for historical docs, negative tests, implementation plans/specs, and intentionally separate Factor Factory artifact contracts.
- `scripts/check_schema_07_browser_acceptance.ps1` and `web/e2e/schema07-diagnostics.spec.ts`: checked browser acceptance artifact for schema `0.7` API-backed workbench behavior using project-local generated Factor Factory runs and ordinary evaluation runs.
- `scripts/check_schema_07_docs_links.py` and `tests/scripts/test_schema_07_docs_links.py`: checked docs link/path validation for current-contract docs and code maps.
- `scripts/check_schema_07_data_scale.py` and `tests/scripts/test_schema_07_data_scale.py`: mandatory read-only current-run scale preflight and comparison gate for rows, combinations, date span, and output footprint, used before Phase 1B and project-internal acceptance commands.
- `scripts/check_schema_07_project_acceptance.ps1` and helper tests if structured helper logic is introduced: checked Slice 11B project-internal acceptance wrapper covering service absence, mandatory data-scale baseline comparison, A-share acceptance, Factor Factory/factor-summary smoke, manifest/display/download validation, private-path leak checks, and path/count summary output.

Retire or rewrite tests:

- `tests/artifacts/test_schema_06_contract.py`: replace with schema `0.7` contract tests or keep only as explicit rejection coverage if a historical fixture is useful.
- API tests that call `/diagnostics/{metric_name}`: rewrite to assert route removal and generic `/tables/{table_key}/preview` behavior.
- `tests/evaluations/test_evaluation_payload.py` and `tests/evaluations/test_public_exports.py`: add or update coverage for internal staging payload boundaries and public exports before diagnostics calculations depend on them.
- Frontend tradability-only tests: preserve any still-valid display-driven behavior, but remove frontend-only diagnostics assumptions.

Ownership note: when a contract-integrator-owned file also appears in a slice file list, the slice treats that file as a dependency surface, not as an independent write target. Slice agents request deltas through the contract integrator before changing registry fields, shared API models, shared TypeScript types, canonical fixtures, or cross-layer payload shapes.

## Slice 1: Schema Registry Source Of Truth

**Goal:** Create a code-owned schema registry that defines the schema `0.7` artifact contract and becomes the source of truth for field counts, grain, unique keys, required-if, nullable-if, sentinel rules, and display eligibility.

**Files:**

- Create `src/quant_research_platform/artifacts/schema_registry.py`
- Create `tests/artifacts/test_schema_07_registry.py`
- Modify `src/quant_research_platform/artifacts/artifact_schema.py` additively only if needed to expose schema `0.7` registry types without flipping current runtime constants.
- Modify `src/quant_research_platform/artifacts/artifact_type_registry.py` additively only if needed for registry discovery; do not switch active runtime outputs in this slice.

**Implementation Steps:**

- [ ] Add immutable dataclasses or Pydantic models for `FieldSpec`, `TableSpec`, `ChartSpec`, `ArtifactSchemaSpec`, and conditional rules, including required `sort_key` and `default_materialized` metadata.
- [ ] Add registry-owned `DiagnosticMetricSpec` for materialized diagnostics before Slice 9 implementation: `policy_family`, exact `metric_name`, `formula_ref`, `unit`, required inputs, optional inputs, canonical optional-input aliases, timing/PIT requirements, status behavior, `default_enabled`, and `deferred` flag. Registry tests must prove the complete N5 v1 metric set is enumerated before metrics are calculated.
- [ ] Add registry/input-bundle-owned `DiagnosticOptionalInputSpec` for optional diagnostics inputs: canonical internal name, allowed aliases, ingress source (`a_share`, `generic_parquet`, evaluation staging, or derived), dtype, timing metadata requirement, privacy redaction behavior, source precedence, column-pruning expectation, and missing behavior.
- [ ] Keep Slice 1 additive: do not change global `ARTIFACT_SCHEMA_VERSION`, writer defaults, loader acceptance, run index behavior, or current exported schema `0.6` constants until Slice 2 migrates writer/loader/index/fixtures together.
- [ ] Register all required minimum vertical-slice tables and metadata files listed above.
- [ ] Register the schema `0.7` time-breakdown generated tables listed in Contract Decisions, including exact paths, display keys, fields, grain, unique keys, sort keys, and display/download eligibility. Do not preserve old `*_summary_by_{year,quarter,month}` display keys.
- [ ] Register conditional table specs with `required=False`, `condition_key`, `display_eligible_when_materialized`, `download_eligible_when_materialized`, `display_when_absent=false`, and an explicit `summary_when_skipped` value.
- [ ] Register `ChartSpec` for `primary_cumulative_return` as an API-derived chart from `portfolio/returns.csv.net_return`, including input series, transform, output series, group/filter rules, preview/payload row limit behavior, and display order.
- [ ] Make registry generate or validate the exact `manifest.artifacts` serialization shape for schema `0.7`; `artifact_schema.py`, `artifact_type_registry.py`, and `manifest_validation.py` are registry adapters, not parallel current-contract sources.
- [ ] Set `default_materialized=false` for every conditional/generated table, not only `portfolio/weights.parquet`.
- [ ] Encode table grain and unique keys:
  - `dimensions/combinations.csv`: grain `combination`, unique key `combination_id`.
  - `metrics/ic.csv`: grain `combination_id + timestamp`, unique key `combination_id,timestamp`.
  - `metrics/grouped_returns.csv`: grain `combination_id + timestamp + group_id`, unique key `combination_id,timestamp,group_id`.
  - `metrics/evaluation_metrics_summary.csv`: grain `combination_id + metric_family + metric_name`, unique key `combination_id,metric_family,metric_name`.
  - `portfolio/returns.csv`: grain `combination_id + timestamp`, unique key `combination_id,timestamp`.
  - `input_quality/coverage_summary.csv`: grain `combination_id + policy_id + input_family + input_name`, unique key `combination_id,policy_id,input_family,input_name`.
  - `dimensions/diagnostic_policies.csv`: grain `policy_id`, unique key `policy_id`.
  - `diagnostics/diagnostic_summary.csv`: grain `combination_id + policy_family + policy_id`, unique key `combination_id,policy_family,policy_id`.
  - `diagnostics/diagnostic_metrics.csv`: grain `combination_id + policy_id + metric_family + metric_name + bucket_id + metric_variant_id`, unique key `combination_id,policy_id,metric_family,metric_name,bucket_id,metric_variant_id`.
  - Conditional/generated files: register the spec-defined generated unique keys for `dimensions/portfolio_assumptions.csv`, `portfolio/weights.parquet`, `diagnostics/tradability_diagnostics.csv`, `diagnostics/tradability_summary.csv`, `diagnostics/portfolio_risk_summary.csv`, `diagnostics/scenario_metrics.csv`, `diagnostics/exposure_capacity_summary.csv`, `diagnostics/diagnostic_observations.parquet`, current year/quarter/month `breakdowns/{period}/evaluation_metrics_summary.csv`, and optional `report.md`.
- [ ] Encode stable `sort_key` for all required and conditional time-series/detail tables; tests must prove sorted writer output or deterministic loader/index comparison uses these registry sort keys rather than table-specific ad hoc ordering.
- [ ] Add first-class field metadata for status columns: `diagnostic_status`, `status_reason`, `required_inputs_met`, and `warning`.
- [ ] Scope status enums by table/field instead of using one global enum blindly: `dimensions/diagnostic_policies.csv.status` must allow the spec-required reserved value `disabled_by_config`; other common `status` / `diagnostic_status` fields keep the spec common enum unless their table spec declares a narrower or wider enum.
- [ ] Add rule primitives for `required_if`, `nullable_if`, and sentinel values such as unavailable numeric metrics when `diagnostic_status != "available"`.
- [ ] Define the complete `manifest.display.tables[]` metadata contract in registry/display specs: `key`, `label`, `role`, `path`, `format`, `section`, `section_order`, `display_order`, `preview.route`, `preview.default_limit`, `preview.max_limit`, `preview.row_count_capability`, `preview.supports_offset`, `download.enabled`, `download.preset_tags`, `dimension_table`, `conditional`, `required`, `empty_state`, `join_key`, `supports_combination_filter`, and optional diagnostic/filter metadata. Keep `display_when_absent` and `summary_when_skipped` as registry-only generated-file semantics used to decide whether a table enters public display.
- [ ] Define the complete `manifest.display.charts[]` metadata contract for derived charts: `key`, `label`, `kind`, `source_table_key`, `x_column`, `input_series`, `transform`, `output_series`, `supports_combination_filter`, `preview.default_limit`, `preview.max_limit`, `display_order`, and empty-state semantics. Chart metadata may reference a registry table source but must not require a materialized cumulative-return artifact.
- [ ] Add registry tests that assert the field count, required/conditional table count, unique display keys, grain, unique key, sort key, `default_materialized`, required-if, nullable-if, and sentinel rule totals are derived from the registry rather than Markdown constants.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/artifacts/test_schema_07_registry.py`

**Acceptance Criteria:**

- Tests can enumerate every schema `0.7` required and conditional table from registry.
- Tests can enumerate schema `0.7` time-breakdown generated table specs and prove old summary-specific breakdown keys are not active.
- Tests can enumerate registry-owned chart specs and prove `primary_cumulative_return` is derived from `portfolio/returns.csv`, not a materialized `portfolio/cumulative_returns.csv`.
- Tests prove `manifest.artifacts` current schema path/group lists come from the registry adapter surface and stale schema `0.6` path builders cannot remain authoritative.
- Tests can enumerate every N5 v1 `DiagnosticMetricSpec` and optional input mapping. Missing metric specs, duplicate metric names within a family, ambiguous aliases, missing timing requirements, or missing status behavior fail before Slice 9 implementation.
- Current schema `0.6` runtime behavior is not changed by Slice 1 alone.
- No test maintains an independent handwritten field-count ledger.
- Registry tests cover stable sort keys and `default_materialized=false` for generated/conditional tables.
- Status enum tests cover the `disabled_by_config` diagnostic policy exception without widening unrelated `status` fields.
- Conditional diagnostics tables are visible to the registry but not required by writer/loader until materialized.
- Conditional absent-file semantics are testable from registry metadata through `display_when_absent` and `summary_when_skipped`.
- Registry can answer whether a table is display/download eligible before a file is generated.

**Risks:**

- Overfitting the registry into a large framework. Keep v1 small: table specs, field specs, validation rules, and display metadata only.
- Drifting display labels into business logic. Use labels for UI only; rules and validation live in field/table specs.

## Slice 2: Metadata Shell, Disabled Diagnostics Producer, And Runtime Switch

**Goal:** Make schema `0.7` metadata and disabled diagnostics writeable, loadable, and indexable through small reviewable gates. Slice 2 is internally split because the runtime switch must not land before disabled diagnostics frames, synthetic writer/loader fixtures, and lightweight index/payload consumers are ready.

**Files:**

- Modify `src/quant_research_platform/artifacts/artifact_writer.py`
- Modify `src/quant_research_platform/artifacts/artifact_loader.py`
- Modify `src/quant_research_platform/artifacts/artifact_schema.py`
- Modify `src/quant_research_platform/artifacts/display_manifest.py`
- Modify `src/quant_research_platform/artifacts/display_tables.py`
- Modify `src/quant_research_platform/artifacts/manifest_validation.py`
- Modify `src/quant_research_platform/artifacts/artifact_comparison.py`
- Modify `src/quant_research_platform/artifacts/artifact_inventory.py`
- Modify `src/quant_research_platform/artifacts/evaluation_comparison_report.py`
- Modify `src/quant_research_platform/evaluations/evaluation_spec.py`
- Modify `src/quant_research_platform/evaluations/period_breakdowns.py`
- Modify `src/quant_research_platform/markets/implementations/a_share/evaluation/evaluation_adapter.py`
- Modify `tests/markets/implementations/a_share/evaluation/test_evaluation_adapter.py`
- Modify `src/quant_research_platform/interfaces/api/run_index.py`
- Modify `src/quant_research_platform/interfaces/api/models.py`
- Modify `src/quant_research_platform/interfaces/api/payloads.py`
- Modify `src/quant_research_platform/interfaces/api/services/comparison_service.py`
- Create `src/quant_research_platform/evaluations/diagnostics/__init__.py`
- Create `src/quant_research_platform/evaluations/diagnostics/spec.py`
- Create `src/quant_research_platform/evaluations/diagnostics/policies.py`
- Create `src/quant_research_platform/evaluations/diagnostics/runner.py`
- Modify `tests/artifacts/artifact_fixtures.py`
- Create `tests/artifacts/test_schema_07_artifacts.py`
- Create `tests/evaluations/diagnostics/test_diagnostics_spec.py`
- Modify `tests/artifacts/test_artifact_contract.py`
- Modify `tests/artifacts/test_evaluation_set_artifacts.py`
- Modify `tests/evaluations/test_time_breakdowns.py`

**Implementation Steps:**

Sub-slice 2A, diagnostics disabled/spec/policy shell:

- [ ] Create the CP1a shared contract surface before diagnostics semantics: `interfaces/api/errors.py`, app-wide handler registration in `interfaces/api/app.py`, Python `ApiErrorEnvelope` helpers, TypeScript `ApiErrorEnvelope` / disabled diagnostics request types, and fixture builders. Slice 5 may migrate route call sites later, but the envelope mechanism and app handlers land here.
- [ ] Implement the minimal diagnostics request/spec/policy shell and disabled diagnostics producer without switching active runtime artifact writes yet.
- [ ] Implement `DiagnosticsRequest` / `DiagnosticsSpec` path needed for disabled diagnostics: missing diagnostics and `enabled=false` with no sibling fields are valid disabled requests; `enabled=false` or missing with non-empty `policy_set_id`, `toggles`, or `materialized_views` is rejected with `error_code="diagnostics_disabled_with_options"`; legacy list/tuple diagnostics shape is rejected with a machine-readable validation error.
- [ ] Implement the shared diagnostics validation error objects behind the common `ApiErrorEnvelope` decision: every diagnostics validation failure must provide `error_code`, human-readable `message`, dotted/bracket `field`, and object `details`; tests must cover single-error and multi-error envelope construction before API routes consume it.
- [ ] Define `DiagnosticPolicy` validation fields required by the spec: `policy_id`, `policy_family`, `identity_scope`, `timing_semantics`, `decision_usable`, `uses_future_return`, `uses_future_information`, `changes_universe`, `changes_signal`, `changes_weights`, `changes_returns`, `status`, `status_reason`, `warning`, `policy_payload_json`, and `missing_input_mode`.
- [ ] Reject v1 post-run diagnostic policies with any `identity_scope` other than `diagnostic_only`; `evaluation_axis` must be routed into `EvaluationSpec` / new-run construction or rejected with a machine-readable error, and `portfolio_policy` / `simulation_policy` remain out of scope.
- [ ] Reject `decision_usable=true` unless `timing_semantics="ex_ante_policy"` and both future-information flags are false; reject any v1 policy where a `changes_*` flag is true.
- [ ] Implement a disabled diagnostics producer in `DiagnosticRunner` before writer integration: `dimensions/diagnostic_policies.csv` gets reserved `policy_id="__diagnostics_disabled__"`, `policy_family="diagnostics_contract"`, `identity_scope="diagnostic_only"`, and `status="disabled_by_config"`; `diagnostics/diagnostic_summary.csv` gets one disabled status row per `combination_id`; `diagnostics/diagnostic_metrics.csv` may be headers-only; no conditional materialized views or optional detail files are generated.

2A tests:

- `conda run -n analysis python -m pytest -q tests/evaluations/diagnostics/test_diagnostics_spec.py -k "disabled or policy or identity_scope"`

2A acceptance:

- Disabled diagnostics request/spec validation works without changing active runtime artifact writes.
- `evaluation_axis`, `portfolio_policy`, `simulation_policy`, future-information misuse, and `changes_*` policies fail with machine-readable errors.
- Diagnostics validation errors have the same envelope fields and path convention later used by API upload/pipeline/schema validation.
- Disabled diagnostics frames follow the reserved policy id and one-summary-row-per-combination invariant.

Sub-slice 2B, schema `0.7` writer/loader synthetic artifact:

- [ ] Add synthetic schema `0.7` fixtures and writer/loader tests before switching runtime producers.
- [ ] Change writer metadata output from `config.json` + `lineage.json` to `run_context.json` for schema `0.7`.
- [ ] Require `summary.json.run_id` and `run_context.json.provenance.run_id`; fail if they differ.
- [ ] Validate exact required `summary.json` fields: `run_id`, `project`, `kind`, `data_stage`, `n_rows_evaluated`, `n_combinations`, `base_evaluation`, `diagnostics`, `artifact_schema_version`, and `warnings`.
- [ ] Validate exact required `run_context.json` fields: `config.evaluation_spec`, `config.diagnostics_spec`, `config.a_share_assumptions`, `config.config_fingerprint`, `provenance.run_id`, `provenance.input_panel_fingerprint`, `provenance.input_data_version`, `provenance.input_artifact_refs`, `provenance.output_artifact_index`, `provenance.diagnostic_policy_fingerprint`, `provenance.generated_at`, `provenance.generator`, `provenance.code_version`, `provenance.environment_summary`, `provenance.staging_payload_fingerprint`, `provenance.factor_definition_version`, and `provenance.random_seed`.
- [ ] For lineage fields that are unavailable in smoke, uploaded-panel, generic Parquet, or deterministic fixtures, require an explicit sanitized `null` / `not_applicable` / `unavailable_reason` value from the registry contract instead of omitting the field.
- [ ] Keep `config.a_share_assumptions` present as a sanitized JSON object for every schema `0.7` run because the spec marks it required; for uploaded-panel, generic Parquet, and non-A-share smoke fixtures, write an explicit non-applicable object rather than omitting the key.
- [ ] Keep portable artifact metadata free of `source_alias`, host, absolute paths, private roots, and runtime URLs.
- [ ] Validate all payload tables against registry before writing.
- [ ] Make the writer consume disabled diagnostics frames from `DiagnosticRunner` rather than duplicating disabled diagnostics semantics inside artifact writing.
- [ ] Build the minimal registry-driven display manifest needed for a schema `0.7` run to load: all materialized required tables, required metadata download item `run_context.json`, and optional `report.md` only when generated and declared.
- [ ] Make loader require `manifest.json`, `summary.json`, `run_context.json`, the minimal valid `manifest.display`, and every required registry table; load `report.md` only when declared by `manifest.artifacts` and `manifest.display.downloads`.

2B tests:

- `conda run -n analysis python -m pytest -q tests/artifacts/test_schema_07_artifacts.py tests/artifacts/test_artifact_contract.py tests/artifacts/test_evaluation_set_artifacts.py -k "writer or loader or manifest or run_context"`

2B acceptance:

- Synthetic schema `0.7` artifacts write and load without flipping current runtime producers.
- Required metadata, required tables, minimal display, run-id equality, lineage completeness, redaction, and optional `report.md` behavior are enforced by loader tests.
- Schema `0.6` runtime constants and producers are still unchanged until 2C.

Sub-slice 2C, runtime switch and adjacent lightweight consumers:

- [ ] Treat Factor Factory/factor-summary active read-path migration as a 2C pre-switch gate, not a later independent checkpoint: do not flip active runtime schema constants until the 2D targeted tests for `_candidate_combination_summary()` and `build_factor_summary.py` pass against schema `0.7` evaluation-set artifacts.
- [ ] Switch active schema `0.7` runtime constants and exported artifact schema names only in this slice, together with writer, loader, fixtures, run index, and payload consumers.
- [ ] Keep 2C scoped to runtime switch plus lightweight list/evidence smoke: direct/ZIP download behavior, selected-download wire shape, upload creation/deletion, full comparison route behavior, and legacy route removal are Slice 5 responsibilities.
- [ ] Make loader and run index reject schema `0.6` explicitly; do not add fallback reads.
- [ ] Make lightweight run index validate table headers from registry without loading full CSV/Parquet tables.
- [ ] Update `RunIndexRecord`, lightweight `RunRecord` list/evidence smoke payloads, artifact inventory, and comparison report inputs to derive fields from `summary.json.run_id`, `run_context.config`, and `metrics/evaluation_metrics_summary.csv` instead of `config.json`, `lineage.json`, `summary["evaluation_id"]`, `ic_summary`, `group_spread_summary`, `performance_summary`, `evaluation_summary`, or `cumulative_returns`.
- [ ] Enforce the schema `0.6` artifact identity decision table in writer/loader/index fixtures: schema `0.7` artifacts must not materialize `metrics/ic_summary.csv`, `metrics/group_spread_summary.csv`, `metrics/performance_summary.csv`, `summaries/evaluation_summary.csv`, `portfolio/cumulative_returns.csv`, old split input-quality files, or old summary-specific breakdown files.
- [ ] Define the display/chart replacement for retired `portfolio/cumulative_returns.csv`: cumulative-return visualization is derived server-side by the API from `portfolio/returns.csv` and exposed as the primary net cumulative chart series, not by a stored schema `0.7` artifact file or frontend inference from truncated preview rows.
- [ ] Migrate `time_breakdowns` in `EvaluationSpec` and `period_breakdowns.py` only to schema `0.7` artifact/table naming, writer/loader fixtures, and retired schema `0.6` summary identities for current year/quarter/month breakdowns. Display validation belongs to Slice 4; preview/direct/ZIP API behavior belongs to Slice 5.
- [ ] Migrate the A-share private evaluation adapter writer path at the same runtime-switch boundary: `_combined_summary`, `_combined_config`, and `_combined_report` must produce schema `0.7` `run_id` / `run_context` semantics, and portable `manifest.json`, `summary.json`, `run_context.json`, and download metadata must not serialize `source_alias` or `evaluation_id`.
- [ ] Make the A-share private evaluation adapter write a schema `0.7` disabled diagnostics artifact in small-fixture tests before the runtime switch is considered complete.
- [ ] Keep optional report and comparison behavior explicit before API visibility: loader, lightweight index, and evidence payloads accept absent display-approved `report.md` as `report: null`, and `comparison.csv` remains an API-runtime artifact only, not a portable `manifest.display` item.
- [ ] Add negative tests for missing required file, missing required table, missing required field, duplicate unique key, invalid schema version, unsafe paths, serialized `source_alias`, missing/minimal invalid display, and run-id mismatch.
- [ ] Add negative tests that a schema `0.7` manifest exposing retired paths or display keys from the artifact identity decision table is rejected.

2C tests:

- `conda run -n analysis python -m pytest -q tests/interfaces/api/test_api.py -k "run_index or evidence or stale or schema"`
- `conda run -n analysis python -m pytest -q tests/evaluations/test_time_breakdowns.py`

2C acceptance:

- Active runtime producers switch to schema `0.7` only after 2A and 2B pass.
- The runtime switch is not complete until Factor Factory/factor-summary active read paths pass targeted schema `0.7` tests and the A-share private adapter proves it can write a schema `0.7` disabled diagnostics artifact; 2C and 2D are reviewed as one switch gate even if their commits are separate.
- Lightweight index, API payload smoke paths, comparison/inventory/report consumers, and current time breakdowns no longer rely on schema `0.6` metadata or summary table identities.
- `time_breakdowns` year/quarter/month outputs remain supported under schema `0.7`.
- A-share adapter small-fixture tests prove the private evaluation path writes schema `0.7` portable metadata without `source_alias` or `evaluation_id`.
- Absent `report.md` produces `report: null` rather than a load/list failure, and comparison remains runtime-only until Slice 5 route/download behavior is implemented.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/artifacts/test_schema_07_registry.py tests/artifacts/test_schema_07_artifacts.py tests/artifacts/test_artifact_contract.py tests/artifacts/test_evaluation_set_artifacts.py`
- `conda run -n analysis python -m pytest -q tests/evaluations/diagnostics/test_diagnostics_spec.py tests/evaluations/test_time_breakdowns.py`
- `conda run -n analysis python -m pytest -q tests/interfaces/api/test_api.py -k "run_index or evidence or stale or schema"`

**Acceptance Criteria:**

- A synthetic schema `0.7` run can be written, loaded, and listed by the lightweight index.
- Disabled diagnostics output follows the reserved policy id and one-summary-row-per-combination invariant exactly.
- The write/load/index sequence is executable in this order: `DiagnosticRunner` emits disabled frames, writer writes required tables and `run_context.json`, minimal display builder exposes required tables/downloads, then loader/index validate.
- A-share, uploaded-panel, and generic Parquet fixtures all include `config.a_share_assumptions`; non-A-share fixtures use an explicit non-applicable sanitized object.
- A schema `0.6` run is excluded or rejected by list/evidence/preview entry points.
- API list/evidence smoke tests use schema `0.7` fixtures without requiring old `config.json`, `lineage.json`, `ic_summary`, `group_spread_summary`, `performance_summary`, `evaluation_summary`, or `cumulative_returns`.
- Schema `0.7` manifests cannot expose `portfolio/cumulative_returns.csv`; cumulative-return charts use `portfolio/returns.csv` as the declared source.
- `time_breakdowns` no longer produces active schema `0.6` summary table identities in schema `0.7`; current year/quarter/month breakdown artifact naming and writer/loader/index smoke behavior are migrated and tested. Display/API/download acceptance is deferred to Slices 4 and 5.
- Runtime-switch smoke tests include the A-share adapter disabled-diagnostics artifact path, absent-report `report: null`, and runtime-only comparison handling.
- Loader and index rejection messages identify the violated contract.
- No schema `0.7` artifact writes separate `config.json` or `lineage.json`.

**Risks:**

- Existing API payload and comparison code may expect `config`, `evaluation_id`, `ic_summary`, `group_spread_summary`, `performance_summary`, `evaluation_summary`, `cumulative_returns`, or retired route families. Keep list/evidence smoke coverage in this slice so those stale assumptions are removed before API route work expands.
- Existing tests may assert `evaluation_id`. Replace those assertions with `run_id` or mark them as historical rejection coverage; schema `0.7` does not keep `evaluation_id` as a serialized alias.

## Slice 2D: Factor Factory Active-Consumer Pre-Switch Gate

**Goal:** Prevent the Slice 2C runtime switch from breaking active Factor Factory and factor-summary consumers by migrating their schema `0.7` read paths before the switch is considered complete.

**Files:**

- Inspect `src/quant_research_platform/research/factor_factory/`
- Inspect `src/quant_research_platform/interfaces/api/routes/factor_factory.py` and related services if present.
- Inspect `web/src/features/factor-factory/` if present.
- Inspect `src/quant_research_platform/markets/implementations/a_share/evaluation/build_factor_summary.py`
- Inspect `src/quant_research_platform/markets/implementations/a_share/evaluation/compare_factor_summaries.py`
- Modify the targeted Factor Factory/factor-summary files needed to make active evaluation-set reads schema `0.7`-ready before the Slice 2C runtime switch is considered complete.

**Implementation Steps:**

- [ ] Run a grep-backed inventory of active Factor Factory, factor-summary, and frontend/API consumers that read evaluation-set artifacts, generated-run evidence, summaries, downloads, or preview keys.
- [ ] Classify each hit as active schema `0.7` migration, separate `factor_factory_run` artifact contract, historical/rejection coverage, or dead/unreachable code.
- [ ] Assign a single implementation owner for `research/factor_factory`, Factor Factory API/frontend tests, `build_factor_summary.py`, and `compare_factor_summaries.py`; other agents may inspect but do not independently edit those files.
- [ ] Migrate active Factor Factory and factor-summary read paths that would break immediately after the Slice 2C runtime switch, not only paths that block Slice 5. This includes `_candidate_combination_summary()` reading `summaries/evaluation_summary.csv` and `build_factor_summary.py` reading `config.json`, old metric summary files, split input-quality files, or `summary["evaluation_id"]`.
- [ ] Preserve `factor_summary.csv` and candidate-summary downstream contracts while changing their evaluation-set artifact inputs to schema `0.7` registry table names.
- [ ] Route any shared schema/API/fixture deltas through the contract integrator before Factor Factory changes proceed.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/interfaces/api/test_factor_factory_api.py tests/research/factor_factory/test_batch_runner.py tests/markets/implementations/a_share/evaluation/test_factor_summary.py`

**Acceptance Criteria:**

- Slice 5 can migrate API preview/download routes without waiting for an unspecified Factor Factory decision.
- Active Factor Factory evaluation-set consumers are migrated enough that they do not turn red immediately after Slice 2C switches runtime schema to `0.7`.
- `batch_runner.py::_candidate_combination_summary()` reads schema `0.7` evaluation summary data from registry tables rather than `summaries/evaluation_summary.csv`.
- `build_factor_summary.py::_build_one_run_summary()` reads schema `0.7` `run_context.json`, metric summaries, input-quality diagnostics, and `summary.run_id` semantics rather than schema `0.6` files or `summary["evaluation_id"]`.
- A schema `0.7` pipeline/evaluation artifact can be consumed by Factor Factory smoke tests and produce stable `candidate_summary.csv` / `factor_summary.csv` outputs.
- `factor_factory_run` keeps its own artifact version/contract unless separately designed.
- Slice 7C may add request-surface or extra proof, but no active Factor Factory/factor-summary code path may still require schema `0.6` evaluation-set artifacts after this gate.

**Risks:**

- Treating Factor Factory as retired would contradict the current decision. This checkpoint preserves it while making its schema `0.7` dependencies explicit.

## Slice 3: Diagnostics Input Bundle And Enabled Policy Runner

**Goal:** Extend the Slice 2 disabled diagnostics contract into enabled diagnostics requests, policy sets, and input-bundle handling without changing base evaluation outputs.

**Files:**

- Modify `src/quant_research_platform/evaluations/diagnostics/__init__.py`
- Modify `src/quant_research_platform/evaluations/diagnostics/spec.py`
- Modify `src/quant_research_platform/evaluations/diagnostics/policies.py`
- Create `src/quant_research_platform/evaluations/diagnostics/inputs.py`
- Modify `src/quant_research_platform/evaluations/diagnostics/runner.py`
- Modify `tests/evaluations/diagnostics/test_diagnostics_spec.py`
- Modify `src/quant_research_platform/evaluations/evaluation_runner.py`
- Modify `src/quant_research_platform/evaluations/evaluation_payload.py`
- Modify `src/quant_research_platform/markets/implementations/a_share/evaluation/evaluation_adapter.py`
- Modify `src/quant_research_platform/markets/implementations/a_share/evaluation/service.py`
- Modify `src/quant_research_platform/markets/implementations/a_share/evaluation/run_factor_evaluation.py`
- Modify `src/quant_research_platform/datasets/parquet_panel.py`
- Modify `src/quant_research_platform/interfaces/api/services/upload_service.py`
- Modify `src/quant_research_platform/interfaces/api/models.py` only through the CP1a/CP3 contract integrator.
- Modify `tests/evaluations/test_evaluation_payload.py`
- Modify `tests/evaluations/test_public_exports.py`

**Implementation Steps:**

- [ ] Extend `DiagnosticsRequest` with enabled policy-set behavior, allowed toggles, and allowed `materialized_views`.
- [ ] Extend `DiagnosticsSpec` as normalized backend contract with allowed policy ids and requested materialized views.
- [ ] Implement `DiagnosticInputBundle` as the only diagnostics input object for the current schema `0.7` base evaluation staging payload plus approved optional processed-panel columns; staging payload paths and private input roots must not become API-visible artifacts or downloadable metadata.
- [ ] Define the internal non-portable evaluation staging payload that diagnostics may consume before artifact writing, including raw frames already available inside evaluation, cost/exposure/neutralization summaries when present, weights/returns, and optional panel columns. This staging object is internal only; portable schema `0.7` artifacts still expose only registry/manifest-approved files.
- [ ] Add optional-column discovery and column pruning for enabled A-share diagnostics paths so diagnostics can see approved columns such as ADV/liquidity, turnover, market cap, industry, benchmark, and tradability fields when present, while missing columns produce unavailable/partial status rather than fabricated values.
- [ ] For generic Parquet diagnostics, require explicit column mapping/spec for market-semantic optional inputs such as tradability, liquidity, industry, benchmark, and regime/control columns. Do not infer market semantics from generic column names alone.
- [ ] Keep uploaded-panel diagnostics disabled in v1: uploaded-panel ingress normalizes only omitted diagnostics or `{enabled:false}`, writes required disabled diagnostics contract files plus sanitized non-applicable A-share assumptions, and does not support enabled uploaded diagnostics or optional-column diagnostics discovery in this plan.
- [ ] Wire optional-column discovery through the enabled real ingress paths in the same slice: A-share adapter/service/CLI and generic Parquet reader. Shared API model shape changes go through CP1a/CP3; uploaded-panel route behavior still lands in Slice 5 as disabled-only.
- [ ] Add `DiagnosticInputBundle` validation for approved input roots, out-of-root `processed_panel_ref`, path traversal, absolute private paths, and sanitized machine-readable error envelopes; raw local paths must not leak into validation errors or artifact metadata.
- [ ] Keep memory bounded: `DiagnosticInputBundle` must select only required base frames and approved optional columns; metadata-only diagnostics must not load full `weights.parquet`, and `diagnostics/diagnostic_observations.parquet` remains disabled by default.
- [ ] Make optional input handling consume `DiagnosticOptionalInputSpec`: canonical names, aliases, source precedence, PIT/timing metadata, privacy redaction, and missing behavior are registry/input-bundle owned, not independently guessed by A-share, generic Parquet, or diagnostics runner code.
- [ ] Implement a default v1 policy set that can emit required disabled/skipped summary rows.
- [ ] Add `DiagnosticRunner` that returns `dimensions/diagnostic_policies.csv`, `diagnostics/diagnostic_summary.csv`, and `diagnostics/diagnostic_metrics.csv` frames with status fields and policy fingerprints.
- [ ] Make `DiagnosticRunner` fail fast when the base evaluation staging payload is missing schema `0.7` required fields, and otherwise use policy `missing_input_mode` for optional diagnostics inputs.
- [ ] Thread optional diagnostics through evaluation writing while preserving existing weights, returns, cumulative returns, and base summaries.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/evaluations/diagnostics/test_diagnostics_spec.py`
- `conda run -n analysis python -m pytest -q tests/evaluations/test_evaluation_payload.py tests/evaluations/test_public_exports.py`
- `conda run -n analysis python -m pytest -q tests/evaluations/test_evaluation_engine_contract.py -k diagnostics`

**Acceptance Criteria:**

- Disabled diagnostics still produce valid required diagnostics files.
- Unknown toggles and unknown materialized views are rejected.
- Policy sets are backend-owned; frontend/API cannot submit arbitrary policy JSON.
- Diagnostics read from `DiagnosticInputBundle`; no diagnostics staging artifact is exposed through artifact metadata or `manifest.display`.
- Tests prove approved optional columns reach diagnostics for A-share and generic Parquet inputs when present, prove bounded column selection when optional columns are absent, and prove uploaded-panel runs stay on the disabled diagnostics path.
- Internal weights/returns may feed diagnostics calculations, but `portfolio/weights.parquet` materialization remains controlled by requested views and registry display/download eligibility.
- Negative tests prove out-of-root/private input refs fail with sanitized errors and no absolute/private paths leak into artifact metadata; API/direct-download/ZIP leakage is covered in Slice 5 after those surfaces are migrated.
- Tests prove diagnostics do not mutate base evaluation returns or weights.

**Risks:**

- Prematurely implementing research calculations. Keep this slice to request normalization, policy metadata, disabled/skipped status, and required frames.

## Slice 4: Registry-Driven `manifest.display`

**Goal:** Expose schema `0.7` required tables and materialized conditional tables through `manifest.display`, with no frontend/backend shortcuts.

**Files:**

- Modify `src/quant_research_platform/artifacts/display_manifest.py`
- Modify `src/quant_research_platform/artifacts/display_tables.py`
- Modify `src/quant_research_platform/artifacts/manifest_validation.py`
- Modify `tests/artifacts/test_artifact_contract.py`
- Modify `tests/artifacts/test_schema_07_artifacts.py`

**Implementation Steps:**

- [ ] Generate display table entries from registry table specs.
- [ ] Assign only spec enum roles: `dimension`, `metric`, `portfolio`, `input_quality`, `summary`, `diagnostic`, and `detail`.
- [ ] For `role=diagnostic` or `role=detail`, require `diagnostic_family`; for other roles, require it to be null or absent.
- [ ] For required diagnostics tables, expose `diagnostic_summary` and `diagnostic_metrics` by default.
- [ ] Register conditional table display metadata but include a table in `manifest.display.tables` only when the artifact path exists and the registry says the table is materialized/display eligible.
- [ ] Use registry absent-file semantics consistently: conditional tables with `display_when_absent=false` stay out of `manifest.display.tables`, while `summary_when_skipped` controls whether skipped/unavailable status appears in summary diagnostics.
- [ ] Use schema `0.7` display/download shape: table entries carry `preview.default_limit`, `preview.max_limit`, `preview.row_count_capability`, `download.enabled`, and `download.preset_tags`; download presets reference `{kind, key}` pairs rather than ambiguous raw paths.
- [ ] Validate the complete table metadata shape from Slice 1 for every display table, including `format`, `section`, `section_order`, `display_order`, `preview.route`, `preview.supports_offset`, `dimension_table`, `conditional`, `required`, `empty_state`, `join_key`, and `supports_combination_filter`.
- [ ] Ensure definition-only dimension tables do not claim `join_key="combination_id"` or `supports_combination_filter=true`.
- [ ] Ensure fact tables that support combination filtering actually include `combination_id`.
- [ ] Ensure display table keys are unique and `display.downloads.items[]` store `RunDisplayDownloadRef={kind,key}` plus label/resolved artifact-path metadata; refs resolve to registry/`manifest.artifacts` run-relative paths, but raw artifact-relative strings are not the public schema `0.7` item identity.
- [ ] Add display/chart metadata for cumulative-return views that points to `portfolio/returns.csv` as the source table; do not expose `portfolio/cumulative_returns.csv` as a table, chart source, direct download, or ZIP item.
- [ ] Add display metadata validation for migrated `time_breakdowns`: year/quarter/month breakdown tables use schema `0.7` registry-backed keys and display metadata, not old `*_summary_by_{period}` display keys.
- [ ] Do not serialize dynamic comparison downloads into portable `manifest.display`; Slice 4 validates only artifact-backed manifest display/download items whose paths resolve through `manifest.artifacts`.
- [ ] Ensure ZIP presets only reference display-approved items.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/artifacts/test_artifact_contract.py tests/artifacts/test_schema_07_artifacts.py -k "display or download or preview"`

**Acceptance Criteria:**

- `manifest.display.schema_version == "0.7"`.
- Required display tables cover all materialized required tables.
- `run_context.json` appears as a required metadata download item when downloads are enabled; optional `report.md` appears only when generated and declared.
- Conditional tables are not downloadable until materialized and declared by display.
- Return/cumulative-return charts are display-backed from `portfolio/returns.csv`; retired `portfolio/cumulative_returns.csv` is rejected if declared.
- Time-breakdown display keys are schema `0.7` registry-backed and do not expose old `*_summary_by_{year,quarter,month}` keys.
- `comparison.csv` is not present in portable `manifest.display`; any comparison download is an API-runtime augmentation added later by Slice 5 when comparison context exists.
- Duplicate display keys, unsafe paths, missing preview limits, missing `empty_state`, missing `preview.route`, missing `preview.supports_offset`, missing section/order metadata, missing `dimension_table`, missing `conditional`, missing `required`, missing `join_key`, missing `supports_combination_filter`, and fake combination filters fail validation.

**Risks:**

- Display builder may accidentally expose non-table metadata as table downloads. Keep metadata download items explicit and table downloads registry-derived.

## Slice 5: API Preview, Download, And Route Migration

**Goal:** Move API evidence, preview, direct download, ZIP download, and run list to schema `0.7`, and remove the legacy diagnostics route entirely.

**Files:**

- Modify `src/quant_research_platform/interfaces/api/routes/runs.py`
- Modify `src/quant_research_platform/interfaces/api/routes/uploads.py`
- Modify `src/quant_research_platform/interfaces/api/routes/artifacts.py`
- Modify `src/quant_research_platform/interfaces/api/routes/downloads.py`
- Modify `src/quant_research_platform/interfaces/api/routes/comparison.py`
- Modify `src/quant_research_platform/interfaces/api/models.py`
- Modify `src/quant_research_platform/interfaces/api/payloads.py`
- Modify `src/quant_research_platform/interfaces/api/services/run_repository.py`
- Modify `src/quant_research_platform/interfaces/api/services/upload_service.py`
- Modify `src/quant_research_platform/interfaces/api/services/table_preview_service.py`
- Modify `src/quant_research_platform/interfaces/api/services/download_service.py`
- Modify `src/quant_research_platform/interfaces/api/services/comparison_service.py`
- Modify `src/quant_research_platform/artifacts/public_metadata.py`
- Modify Factor Factory API routes/services if they expose generated run evidence, summaries, or downloads.
- Modify `src/quant_research_platform/artifacts/artifact_package.py`
- Modify `tests/interfaces/api/test_api.py`
- Modify `tests/interfaces/api/test_run_comparison.py`
- Modify or add Factor Factory API tests that cover generated-run schema `0.7` evidence/download behavior if those surfaces exist.
- Modify `tests/interfaces/api/artifact_run_fixtures.py`

**Implementation Steps:**

- [ ] Remove `diagnostic_table()` and its route decorator for `/diagnostics/{metric_name}`.
- [ ] Define shared API models in `interfaces/api/models.py` for `ApiErrorEnvelope`, schema `0.7` `UploadedPanelCreateRequest`, `UploadedPanelCreateResponse`, display download refs, and preview payloads before route handlers are migrated; route code must not define local incompatible shapes. `DiagnosticsOptions` is deferred to Slice 7B/CP4 after Slice 8 requestability wiring.
- [ ] Add `interfaces/api/errors.py` and register app-wide local API handlers in `interfaces/api/app.py` so `HTTPException`, request validation errors, diagnostics validation errors, artifact validation errors, and route-level project validation failures return the common `ApiErrorEnvelope`. This applies to Factor Factory routes too if they are mounted in the same local API app; update their tests rather than adding a second envelope.
- [ ] Keep Factor Factory's `factor_factory_run` artifact schema `0.1` contract separate, but migrate local API error responses for Factor Factory routes to the same app-wide `ApiErrorEnvelope` and update `tests/interfaces/api/test_factor_factory_api.py` string-detail assertions accordingly.
- [ ] Return the common error envelope for Pydantic validation, diagnostics policy validation, pipeline validation, upload validation, schema/manifest validation, and download item validation; add status-code tests for `400`, `422`, `404`, and `409` paths.
- [ ] Remove or reject legacy path-inference preview routes `/metrics/{metric_name}`, `/portfolio/{metric_name}`, and `/breakdowns/{group}/{metric_name}` for schema `0.7` runs; no display-key alias exception is allowed.
- [ ] Add tests that `/api/runs/{source_alias}/{run_id}/diagnostics/{metric_name}` returns 404; add matching tests for every retired path-inference route.
- [ ] Keep `/api/runs/{source_alias}/{run_id}/tables/{table_key}/preview` as the only diagnostics preview route.
- [ ] Make evidence payload include runtime `preview_url` only from display table keys.
- [ ] Make evidence payload include API-derived chart payloads keyed by registry `ChartSpec` keys, including `primary_cumulative_return`; frontend must not derive cumulative returns from preview rows.
- [ ] Update loader, run index, evidence route, and payload models so a valid schema `0.7` run without optional `report.md` returns successfully with `report: null` and an explicit report-unavailable state; status and warnings must come from `summary.json` plus `diagnostics/diagnostic_summary.csv`, not from report prose.
- [ ] Update the standalone `/report.md` route to return 404 when `report.md` is absent or not display-approved; do not recreate a compatibility report file.
- [ ] Make schema `0.7` direct artifact download resolve by `RunDisplayDownloadRef={kind,key}` only. If `GET /api/runs/{source_alias}/{run_id}/artifacts/{artifact_item:path}` remains mounted, it must first parse/resolve a display ref or reject schema `0.7` raw paths; add positive coverage for `metadata_file:run_context` and negative coverage for retired `config.json` / `lineage.json`, raw paths, and table-vs-metadata key collisions.
- [ ] Make direct artifact downloads require display-approved `download.enabled` items from `download_service.py`; remove direct `config.json` / `lineage.json` branches and migrate metadata downloads to `run_context.json`.
- [ ] Update `public_metadata.py` and API tests so `run_context.json` redaction is verified for writer output, direct metadata download, and selected ZIP; retire old `config.json` / `lineage.json` redaction expectations.
- [ ] Migrate uploaded-panel `POST /api/runs` creation through `routes/uploads.py` and `upload_service.py` to exact schema `0.7` request/response models: frontend omits `diagnostics`; backend also accepts `{enabled: false}` for API callers; enabled diagnostics or sibling diagnostics options are rejected with the common envelope; response includes `source_alias`, `run_id`, `artifact_schema_version="0.7"`, `data_stage="uploaded_panel"`, and the route/evidence fields the frontend uses.
- [ ] Add schema `0.7` uploaded-panel deletion tests for the local cleanup exception: positive deletion requires `data_stage="uploaded_panel"`, run directory as an immediate child of a configured root, and `manifest.json`; negatives cover generated/private runs, missing manifest, unknown root, out-of-root path, unsafe run id, duplicate `(source_alias, run_id)` across roots, and symlink/junction run directories whose resolved path escapes the configured root. Assert outside targets remain untouched.
- [ ] Add malicious `processed_panel_ref` / private-path fixtures through API evidence, direct download, and selected ZIP paths to prove private inputs do not leak after download surfaces are migrated.
- [ ] Define and implement the schema `0.7` selected-download wire shape before route migration: JSON `POST` bodies accept `items: [{"kind": "...", "key": "..."}]`; browser `GET` links use repeated compact refs such as `items=table:diagnostic_summary&items=metadata_file:run_context` and, for separate comparison actions only, `items=dynamic:comparison`. Reject old raw string ids and unregistered kind aliases such as `metadata` in schema `0.7` request models.
- [ ] Make selected ZIP packages validate item membership through `manifest.display.downloads`, including `{kind, key}` preset item resolution and stable cache fingerprints based on `(kind, key)` identity.
- [ ] Make comparison routes/services consume schema `0.7` `run_context` and `evaluation_metrics_summary`, and preserve `(source_alias, run_id)` in every comparison row/payload.
- [ ] Migrate existing comparison context/download behavior so `comparison.csv` is generated only as API-runtime `{kind: "dynamic", key: "comparison"}` when comparison context exists; route filters, context service, ZIP dynamic item handling, frontend fixtures, and download picker tests must not treat it as a static artifact file or serialize it into `manifest.display`.
- [ ] Ensure preview limits use display metadata and enforce hard max limits for CSV and Parquet.
- [ ] Add summary-card aggregation contract/tests: API and frontend summary cards derive only from `summary.json`, `metrics/evaluation_metrics_summary.csv`, and `diagnostics/diagnostic_summary.csv`; never from retired summary tables or truncated preview rows.
- [ ] Add positive API/download coverage for migrated `time_breakdowns`: year/quarter/month breakdown tables appear only through `manifest.display`, preview through `/tables/{table_key}/preview`, and direct/ZIP downloads through display-approved `{kind, key}` items; `/breakdowns/{group}/{metric_name}` is rejected.
- [ ] Add positive API evidence/chart coverage for cumulative-return display derived from `portfolio/returns.csv`, and negative coverage that `portfolio/cumulative_returns.csv` cannot be previewed or downloaded in schema `0.7`.
- [ ] Use registry `display_when_absent` / `summary_when_skipped` in API evidence and preview payloads so absent conditional files are hidden while skipped/unavailable summaries remain visible only when the registry says they should.
- [ ] Ensure run list/evidence skip or reject schema `0.6` runs without weakening loader validation.
- [ ] Preserve `(source_alias, run_id)` across route resolution, preview URLs, direct downloads, selected ZIP cache fingerprints, and frontend route state.
- [ ] Inspect and migrate Factor Factory generated-run API consumers in this slice if they read schema `0.6` metadata, report-only status, hard-coded preview paths, or old download item ids.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/interfaces/api/test_api.py tests/interfaces/api/test_run_comparison.py`
- `conda run -n analysis python -m pytest -q tests/interfaces/api/test_factor_factory_api.py`
- `conda run -n analysis python -m pytest -q tests/artifacts/test_artifact_package.py`

**Acceptance Criteria:**

- API exposes diagnostics only through display table preview and display-approved downloads.
- Legacy diagnostics route is absent; no compatibility wrapper infers `diagnostics/{metric_name}.csv`.
- Retired path-inference preview routes return 404, 410, or 422 for schema `0.7`; only `/tables/{table_key}/preview` can preview tables.
- Evidence payload succeeds for a valid schema `0.7` run that omits optional `report.md`.
- Direct and ZIP downloads include `run_context.json` and exclude retired schema `0.6` metadata assumptions.
- `run_context.json` redaction and private-path leak tests cover API evidence, direct download, and selected ZIP.
- Uploaded-panel creation produces schema `0.7` artifacts with disabled diagnostics, `data_stage="uploaded_panel"`, required diagnostics contract files, sanitized non-applicable A-share assumptions, and the exact shared create response shape.
- Uploaded-panel deletion remains narrowly gated to safe local uploaded runs and rejects generated/private/out-of-root/duplicate/symlink-junction escape identities.
- Old raw-string selected-download items are rejected; `{kind, key}` item expansion, preset expansion, direct download resolution, and ZIP cache fingerprints have positive/negative coverage.
- Dynamic comparison download behavior is explicit: `comparison.csv` is available only as API-runtime `{kind: "dynamic", key: "comparison"}` when comparison context exists, never as a portable manifest item.
- Comparison API works from schema `0.7` fixtures and does not read `config.json`, `lineage.json`, `ic_summary`, `group_spread_summary`, `performance_summary`, `evaluation_summary`, or `cumulative_returns`.
- Summary-card aggregation is API/frontend-tested against canonical schema `0.7` sources and not retired summary files.
- Time breakdown API/download behavior is display-backed; retired `/breakdowns/{group}/{metric_name}` is rejected.
- Preview responses remain bounded and include `has_more` / `total_rows_exact` where applicable.

**Risks:**

- Removing the route is coupled to frontend migration in Slice 6. Treat Slices 5 and 6 as one integration checkpoint for broad verification; do not merge route removal before the generic frontend diagnostics and download consumers are updated and tested.

## Slice 6: Minimal Generic Frontend Diagnostics Display

**Goal:** Render diagnostics tables generically from `manifest.display` with no table-name inference, no diagnostics recalculation, and no special-case dependency on old tradability UI.

**Files:**

- Create `web/src/features/evidence/DiagnosticTables.tsx`
- Create `web/src/features/evidence/__tests__/DiagnosticTables.test.tsx`
- Modify `web/src/shared/api/api.ts`
- Modify `web/src/shared/testFixtures.ts`
- Modify `web/src/features/evidence/ResultsView.tsx`
- Modify `web/src/features/evidence/useRunEvidence.ts`
- Modify `web/src/features/evidence/runEvidenceState.ts`
- Modify `web/src/features/evidence/CsvBrowser.tsx`
- Modify `web/src/features/evidence/ArtifactDownloads.tsx`
- Modify `web/src/features/evidence/useArtifactDownloads.ts`
- Modify `web/src/shared/downloads.ts`
- Modify `web/src/shared/api/api.test.ts`; create it if it is missing.
- Modify `web/src/features/evidence/__tests__/downloads.test.ts` if present.
- Modify `web/src/features/uploads/runForm.ts`
- Modify `web/src/features/uploads/useCreateRun.ts`
- Modify `web/src/features/uploads/CreateRunPanel.tsx`
- Modify `web/src/features/uploads/__tests__/CreateRunPanel.test.tsx`
- Inspect frontend Factor Factory tests and add targeted schema `0.7` coverage if an active frontend consumer exists; otherwise record grep-backed evidence that no active frontend Factor Factory consumer reads retired evaluation-set artifacts.
- Modify `web/src/features/evidence/__tests__/useRunEvidence.test.tsx`
- Modify `web/src/features/evidence/__tests__/runEvidenceState.test.ts`
- Modify `web/src/shell/App.test.tsx`

**Implementation Steps:**

- [ ] Add TypeScript fields for schema `0.7` evidence display table roles, preview descriptors, API-derived chart payloads, `ApiErrorEnvelope`, display/download refs, and schema `0.7` uploaded-panel create request/response. Do not add `DiagnosticsOptions` in Slice 6; those land in Slice 7B/CP4 and are refreshed for Pipeline Builder in Slice 7D/CP5.
- [ ] Land `7D0 uploaded-panel disabled UX` in the Slice 5/6 integration checkpoint: uploaded-panel forms/hooks hide diagnostics controls, omit diagnostics by default, accept only backend `{enabled:false}` semantics when needed, render common error-envelope failures, and update delete UX tests against schema `0.7` uploaded runs.
- [ ] Split evaluation-set download types from Factor Factory download types: use an evaluation schema `0.7` display item reference such as `RunDisplayDownloadRef = {kind, key}` while preserving a separate `FactorFactoryDownloadPreset` / `factor_factory_run` type with `items: string[]` unless that contract is separately versioned.
- [ ] Migrate evaluation display download preset item types from raw `string[]` ids to `{kind, key}[]` references, including API models, shared TypeScript types, fixture builders, picker default state, selected-item toggles, URL builders, selected ZIP request builders, and tests.
- [ ] Build generic diagnostics sections from `display.tables` where `role` is `diagnostic` or `detail`; do not use plural legacy role/group names such as `diagnostics`.
- [ ] Use table label, status, and preview URL from display/API; do not infer file paths or recompute metrics.
- [ ] Render return/cumulative-return chart state from API-provided derived chart payloads declared by display `ChartSpec`; frontend fixtures must not include `portfolio/cumulative_returns.csv` as a schema `0.7` downloadable table or require `cumulative_net_return` in table preview rows.
- [ ] Render summary cards only from API payload fields derived from `summary.json`, `metrics/evaluation_metrics_summary.csv`, and `diagnostics/diagnostic_summary.csv`; never infer card values from retired summary files or truncated preview rows.
- [ ] Keep combination switching behavior only for display tables with `supports_combination_filter=true`.
- [ ] Do not infer "latest" rows from truncated or non-exact preview counts.
- [ ] Keep CSV/detail preview default-collapsed and bounded by API-provided preview limits.
- [ ] Use API/display `empty_state` plus registry absent-file semantics instead of inventing frontend-only messages for missing conditional diagnostics views.
- [ ] Add unit coverage for `report: null`: the frontend renders the API/display empty state, does not show stale "select report.md" wording, and excludes report from the download picker when no display-approved report item exists.
- [ ] Preserve download picker parity with `manifest.display.downloads`.
- [ ] Remove or demote `TradabilityDiagnostics` if it depends on old route/table assumptions; any retained tradability card must consume generic preview payloads and display metadata.
- [ ] Add a negative frontend test that no diagnostics UI/state depends on the hard-coded `tradability_diagnostics` key unless `manifest.display.tables[]` declares that exact table.
- [ ] Add a frontend duplicate-run-id test fixture where two runs share the same `run_id` under different `source_alias` values; verify run switching, evidence cache keys, preview URLs, selected download state, and download URLs remain scoped by `(source_alias, run_id)`.
- [ ] Migrate shell-level `App.test.tsx` expectations from old `/metrics` routes, `performance_summary` display keys, and `metrics/performance_summary.csv` direct downloads to schema `0.7` display keys, `/tables/{table_key}/preview`, and `{kind, key}` download identity; any retained schema `0.6` cases must be marked historical/rejection coverage.
- [ ] Add shared API fixture/type tests for common error envelopes, schema `0.7` evidence display payloads, chart source metadata, and selected-download `{kind, key}` refs so frontend consumers cannot drift from backend models.

**Tests:**

- `Push-Location web; try { npm test -- --run --maxWorkers=1 } finally { Pop-Location }`
- Narrow first if needed: `Push-Location web; try { npm test -- --run --maxWorkers=1 src/features/evidence/__tests__/DiagnosticTables.test.tsx src/features/evidence/__tests__/useRunEvidence.test.tsx src/features/evidence/__tests__/downloads.test.ts src/features/uploads/__tests__/CreateRunPanel.test.tsx src/shared/api/api.test.ts src/shell/App.test.tsx } finally { Pop-Location }`

**Acceptance Criteria:**

- Frontend renders required diagnostics summary/metrics tables from display metadata.
- Frontend does not call `/diagnostics/{metric_name}`.
- Frontend does not call retired `/metrics`, `/portfolio`, or `/breakdowns` preview routes for schema `0.7` runs.
- Combination switching refreshes only combination-aware diagnostics tables.
- Return/cumulative-return chart rendering does not depend on a materialized `portfolio/cumulative_returns.csv` file.
- Frontend renders API-derived chart payloads and does not require `cumulative_net_return` to appear in `portfolio/returns.csv` preview rows.
- Summary cards are rendered from canonical schema `0.7` payload fields, not retired schema `0.6` summary files or preview rows.
- Uploaded-panel create/delete UX works against schema `0.7` request/response types with diagnostics hidden/disabled and common error-envelope rendering.
- `report: null` unit coverage proves no stale report-selection wording and no report download item appears when optional `report.md` is absent.
- Download picker and "select all" use `{kind, key}` display item identity, include every display-approved diagnostics table, and include no hidden diagnostics items.
- Factor Factory frontend types do not inherit evaluation-set `{kind, key}` download refs unless its own `factor_factory_run` contract is separately migrated.
- Duplicate `run_id` frontend fixtures prove no cache, route, preview, or download state drops `source_alias`.

**Risks:**

- A generic table UI can become noisy. Keep first slice minimal: role-grouped sections plus existing detail preview and downloads.

## Slice 7: Pipeline, CLI, And Request Surface Migration Split Gates

**Goal:** Replace old tuple/list diagnostics semantics with a validated diagnostics request object across Pipeline Builder, snapshots, executable pipeline runs, CLI/config, A-share evaluation entry points, generic Parquet evaluation, and active Factor Factory consumers. This slice is split so shared contracts and conditional-view registration cannot block or conflict.

**Files:**

- Modify `src/quant_research_platform/pipelines/a_share/config.py`
- Modify `src/quant_research_platform/pipelines/a_share/pipeline_step_registry.py`
- Modify `src/quant_research_platform/interfaces/api/services/pipeline_service.py`
- Modify `src/quant_research_platform/interfaces/api/routes/pipeline.py`
- Modify `src/quant_research_platform/interfaces/cli/commands.py`
- Modify `src/quant_research_platform/datasets/parquet_panel.py`
- Modify `src/quant_research_platform/markets/implementations/a_share/evaluation/run_factor_evaluation.py`
- Modify `src/quant_research_platform/markets/implementations/a_share/evaluation/service.py`
- Modify `src/quant_research_platform/markets/implementations/a_share/evaluation/build_factor_summary.py`
- Modify `src/quant_research_platform/research/factor_factory/batch_runner.py`
- Modify Factor Factory backend request/run surfaces if they construct evaluation specs, generated runs, or artifact evidence.
- Modify `web/src/shared/api/api.ts`
- Modify `web/src/features/pipeline/PipelineBuilder.tsx`
- Modify Factor Factory frontend request/run surfaces if they construct diagnostics/evaluation payloads or read generated-run evidence.
- Modify `tests/pipelines/a_share/test_pipeline_config.py`
- Modify pipeline runner / step registry tests that assert evaluation step output files.
- Modify `tests/interfaces/api/test_pipeline_service.py`
- Modify `tests/interfaces/api/test_factor_pipeline.py`
- Modify `tests/interfaces/api/test_factor_factory_api.py`
- Modify `tests/markets/implementations/a_share/evaluation/test_factor_summary.py`
- Modify or add generic Parquet adapter / `evaluate-parquet` CLI tests covering schema `0.7` disabled diagnostics output.
- Modify `tests/research/factor_factory/test_batch_runner.py`
- Modify `tests/interfaces/cli/test_commands.py`
- Modify `web/src/features/pipeline` tests if present or add coverage in `web/src/shared/api/api.test.ts`

**Merge Order:**

- 7A backend diagnostics request contract for snapshots/runs, disabled/no-view only, after Slice 2A and before conditional options exposure.
- Slice 8 conditional view requestability gate.
- 7B A-share pipeline, generic Parquet adapter, CLI/config, and diagnostics options discovery, after Slice 8 provides registered vs requestable view metadata.
- 7C Factor Factory and factor-summary schema `0.7` migration; no retirement in Phase 1.
- 7D frontend Pipeline Builder diagnostics-options migration only after Slice 10 completes implemented/requestable conditional-view metadata and the contract integrator refreshes `DiagnosticsOptions`; do not let Pipeline Builder consume provisional registered-only view metadata. Uploaded-panel disabled UX already lands as `7D0` in the Slice 5/6 integration checkpoint.

**Implementation Steps:**

- [ ] 7A: Remove `PipelineConfig.diagnostics: tuple[str, ...]` and add `diagnostics_request` carrying the normalized object shape for disabled/no-view requests only.
- [ ] 7A: Reject legacy list/tuple diagnostics payloads with the common `ApiErrorEnvelope` shape, including `error_code`, `message`, dotted/bracket `field`, and object `details`.
- [ ] Snapshot endpoint may validate and persist diagnostics config, but must not run diagnostics, write schema `0.7` artifacts, or enter run index.
- [ ] 7B: Update A-share pipeline step registry/checkpoint contracts from `evaluation/config.json` to schema `0.7` `evaluation/run_context.json` and matching required output files.
- [ ] 7B: Executable pipeline runs normalize diagnostics then pass the spec into the shared evaluation/diagnostics writer path.
- [ ] 7B: CLI/config parsing constructs the same `DiagnosticsSpec`; it must not implement a separate diagnostics computation path.
- [ ] 7B: Update the generic `evaluate-parquet` path and `datasets/parquet_panel.py` so absent diagnostics and `enabled=false` normalize through the same disabled `DiagnosticsSpec` as A-share/uploaded-panel paths; add a CLI test proving `evaluate-parquet` writes schema `0.7` diagnostics-disabled required files.
- [ ] 7B: Define `DiagnosticsOptions` response/shared type with exact fields `default_policy_set_id`, `policy_sets`, `toggles`, `materialized_views`, `default_enabled`, `requestable`, and `implemented`; include registered-but-unimplemented conditional views as `requestable=false, implemented=false` in the API/shared type, while Pipeline Builder hides non-requestable views by default.
- [ ] 7B: Expose `policy_set_id`, allowed diagnostics toggles, allowed materialized views, and defaults through the existing pipeline options/configuration surface or shared API type; do not add a phase-specific diagnostics endpoint.
- [ ] 7C: Migrate Factor Factory `_candidate_combination_summary()` in `src/quant_research_platform/research/factor_factory/batch_runner.py` from `summaries/evaluation_summary.csv` to schema `0.7` registry table names such as `metrics/evaluation_metrics_summary.csv`.
- [ ] 7C: Migrate `build_factor_summary.py` from schema `0.6` run discovery and `config.json` / `metrics/ic_summary.csv` / `metrics/group_spread_summary.csv` / `metrics/performance_summary.csv` / `input_quality/label_availability.csv` / `summary["evaluation_id"]` reads to schema `0.7` `run_context.json`, `input_quality/coverage_summary.csv`, and registry tables.
- [ ] 7C: Keep `compare_factor_summaries.py` and `factor_summary.csv` stable as the downstream comparison CLI contract; update targeted tests for migrated schema `0.7` factor-summary outputs.
- [ ] Keep `factor_factory_run` artifacts on their existing artifact schema/version unless separately designed; do not force Factor Factory's own display/download preset shape to evaluation-set `{kind, key}` as part of this migration.
- [ ] Inspect other Factor Factory generation paths and migrate any diagnostics/evaluation request or generated-artifact consumer to the same `DiagnosticsSpec` and schema `0.7` artifact contract; if no additional active consumer exists, add a checked static assertion or targeted test proving active Factor Factory paths do not read retired evaluation-set files such as `config.json`, `lineage.json`, `summaries/evaluation_summary.csv`, `metrics/ic_summary.csv`, `metrics/group_spread_summary.csv`, `metrics/performance_summary.csv`, or `portfolio/cumulative_returns.csv`.
- [ ] 7D: After Slice 10/CP5, Pipeline Builder consumes only backend/shared `DiagnosticsOptions` values and rejects frontend-only options not present in that surface; it must not submit arbitrary policy JSON.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_pipeline_config.py tests/interfaces/api/test_pipeline_service.py tests/interfaces/api/test_factor_pipeline.py tests/interfaces/api/test_factor_factory_api.py tests/markets/implementations/a_share/evaluation/test_factor_summary.py tests/research/factor_factory/test_batch_runner.py tests/interfaces/cli/test_commands.py`
- Add or include the narrow generic Parquet adapter test proving `evaluate-parquet` writes schema `0.7` diagnostics-disabled required files.
- `Push-Location web; try { npm test -- --run --maxWorkers=1 src/shared/api/api.test.ts } finally { Pop-Location }`

**Acceptance Criteria:**

- All entry points converge to the same `DiagnosticsSpec`.
- Disabled diagnostics with options is rejected consistently.
- Unknown toggles/views are rejected consistently.
- Diagnostics options discovery has a single backend/TypeScript contract and does not advertise registered-but-unimplemented views as requestable.
- Snapshot remains configuration-only; executable run continues through evaluation.
- Pipeline step registry no longer names retired schema `0.6` output files such as `evaluation/config.json`.
- Generic `evaluate-parquet` writes schema `0.7` artifacts with diagnostics-disabled required files.
- Factor Factory active evaluation-set consumers are migrated to schema `0.7`; no Factor Factory or factor-summary CLI retirement occurs without separate approval.
- A-share factor-summary tooling is migrated to schema `0.7` and keeps `factor_summary.csv` stable.
- Factor Factory generated evaluation summaries no longer read `summaries/evaluation_summary.csv`, while `factor_factory_run` artifacts keep their own versioned contract.
- Factor Factory active-path migration is proven by targeted tests; inactive/non-applicable paths are grep/static-checked only to ensure they do not read retired evaluation-set artifacts.
- `compare_factor_summaries.py` remains compatible with `factor_summary.csv` output after schema `0.7` migration; retirement is out of scope and requires a separate approved plan.

**Risks:**

- A-share pipeline tests may still rely on old `industry_neutrality` tuple behavior. Replace with explicit v1 diagnostics request tests, not a shim.

## Slice 8: Conditional View Requestability Gate

**Goal:** Wire requestability for conditional diagnostics views already registered by Slice 1, without redefining registry paths, columns, unique keys, display keys, or eligibility.

**Files:**

- Modify `src/quant_research_platform/evaluations/diagnostics/policies.py`
- Modify `src/quant_research_platform/evaluations/diagnostics/runner.py`
- Modify `src/quant_research_platform/evaluations/diagnostics/inputs.py`
- Modify `tests/artifacts/test_schema_07_registry.py`
- Modify `tests/evaluations/diagnostics/test_diagnostics_spec.py`

**Implementation Steps:**

- [ ] Reuse Slice 1 conditional table registry entries; do not redefine artifact paths, columns, unique keys, sort keys, display keys, default materialization, or display/download eligibility in this slice.
- [ ] Any discovered registry field/path/default-materialization change must go back to the contract integrator before Slice 8 continues.
- [ ] Split registry knowledge from requestability: unimplemented conditional views may be registered for contract metadata, but they must not be advertised in allowed `DiagnosticsSpec.materialized_views` or frontend/backend options.
- [ ] Include `portfolio/weights.parquet` in the generated file registry with unique key `combination_id,timestamp,asset_id`, default materialized false, and display/download eligibility only when generated.
- [ ] Require `DiagnosticPolicy.output_tables` to allow a conditional view.
- [ ] Require `DiagnosticsSpec.materialized_views[]` to request a conditional view.
- [ ] If a registered-but-unimplemented conditional view is requested, reject it with `error_code="diagnostic_view_not_implemented"`; use skipped/unavailable summary rows only for configured-off or missing-input cases on implemented views.
- [ ] Ensure unmaterialized conditional tables are absent from `manifest.artifacts`, `manifest.display.tables`, downloads, and ZIP presets.
- [ ] Add negative tests for unknown view, disallowed view for policy, declared display table without file, and generated file without registry declaration.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/artifacts/test_schema_07_registry.py tests/evaluations/diagnostics/test_diagnostics_spec.py tests/artifacts/test_schema_07_artifacts.py -k "conditional or materialized or display"`

**Acceptance Criteria:**

- Conditional table path/key/status/display eligibility is reviewable from registry and tests.
- Slice 8 changes requestability and policy wiring only; registry path/key/field ownership remains in Slice 1 and contract-integrator checkpoints.
- No conditional view is visible to API/frontend/download until materialized.
- Registered-but-unimplemented views are not requestable and cannot silently degrade into skipped/unavailable output.
- Missing optional exposure/capacity fields produce warning/unavailable status rows, not silent empty tables.

**Risks:**

- Treating registry-only entries as user-visible commitments. Keep display/download eligibility false until materialization.

## Slice 8B: Data Scale And Resource Preflight

**Goal:** Establish the current generated-run scale baseline before Phase 1B and project-internal acceptance so diagnostics tests do not exceed the user's memory/resource boundary.

**Files:**

- Add `scripts/check_schema_07_data_scale.py`.
- Add `tests/scripts/test_schema_07_data_scale.py`.
- Do not commit generated private outputs or baseline JSON from ignored run roots.

**Implementation Steps:**

- [ ] Inspect only configured project-local run roots and existing generated runs; do not start API/Vite services and do not run factor/evaluation pipelines in this preflight.
- [ ] Record maximum rows, combinations, date span, table count, output footprint, download/ZIP footprint, frontend preview row footprint, and observed runtime/memory proxy where available into ignored baseline artifact `reports/private_real/schema_07_acceptance/data_scale_baseline.json`.
- [ ] Define the baseline JSON schema and helper modes: `--write-baseline`, `--compare-run-root`, and `--fail-if-missing`.
- [ ] Add a checked comparison mode so Slice 9, Slice 10, and Slice 11B runs must prove they stay at or below the baseline unless separately approved; do not rely on a manual checklist for this gate.
- [ ] Make `scripts/check_schema_07_project_acceptance.ps1` call this helper or equivalent checked logic and fail closed when rows, combinations, date span, or output footprint exceed the recorded baseline.
- [ ] Fail closed when `diagnostics/diagnostic_observations.parquet`, selected ZIP/download outputs, or frontend preview/render footprints exceed baseline limits or cannot be classified.
- [ ] Report paths/counts only; do not print private row-level data or absolute private input roots.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/scripts/test_schema_07_data_scale.py`

**Acceptance Criteria:**

- Phase 1B implementers know the concrete current-run scale before data tests start.
- Later project-internal data tests have a non-ambiguous, machine-checked baseline for rows, combinations, date span, output footprint, download/ZIP footprint, frontend preview footprint, and runtime/memory proxy where available.
- 11B project-internal acceptance fails closed if the scale helper is absent, unreadable, or cannot classify the baseline/result comparison.

**Risks:**

- A baseline tool can accidentally become a private data dumper. Keep output to aggregate counts, sizes, and project-relative run paths.

## Slice 9: Phase 1B V1 Materialized Diagnostics Calculations

**Goal:** Complete N5 v1 low-cardinality post-run diagnostics calculations after the Phase 1A schema/API/display vertical slice is stable.

**Files:**

- Modify `src/quant_research_platform/evaluations/diagnostics/runner.py`
- Add focused helper modules under `src/quant_research_platform/evaluations/diagnostics/` only when a calculation becomes too large for the runner.
- Modify `tests/evaluations/diagnostics/`
- Modify `tests/evaluations/test_evaluation_engine_contract.py`

**Implementation Steps:**

- [ ] Split implementation into independently reviewable sub-slices:
  - 9A common metrics emitter, status semantics, registry-backed field validation, and bounded input access.
  - 9B risk/path diagnostics from returns only.
  - 9C cost diagnostics from modeled cost/component fields only.
  - 9D capacity/liquidity proxy diagnostics from ADV/liquidity/weights/trade-value inputs.
  - 9E regime/scenario/control diagnostics from project-local regime, benchmark, market segment, and control columns.
  - 9F A-share assumption/tradability/neutralization diagnostics rows.
- [ ] Implement N5 v1 `diagnostics/diagnostic_metrics.csv` families that can be computed from existing base evaluation payloads and project-local optional columns.
- [ ] Add `diagnostic_metrics.csv` fields from the registry/spec field list, including `policy_family`, `metric_name`, `metric_value`, `unit`, `diagnostic_status`, `status_reason`, `required_inputs_met`, and `warning`; keep `diagnostic_family` only as `manifest.display` metadata for `role=diagnostic` / `role=detail` tables, not as a `diagnostic_metrics.csv` column.
- [ ] Implement only metrics declared in registry-owned `DiagnosticMetricSpec`. Each metric row must have exact `metric_name`, formula reference, unit, window/annualization/quantile convention where applicable, required/optional inputs, timing/PIT requirement, and missing/partial status behavior.
- [ ] Use registry `required_if` / `nullable_if` rules for family-specific fields instead of sparse required columns.
- [ ] Add path/risk metrics, including volatility, drawdown, tail proxy, and ES proxy rows where the required return series exists; otherwise emit unavailable/warning rows.
- [ ] Add cost diagnostics from existing return/cost fields and A-share assumptions: modeled tax/commission/transfer/slippage/impact components, break-even rows, and sensitivity rows where inputs exist; never infer unmodeled costs as zero.
- [ ] Add capacity proxy diagnostics from existing ADV/liquidity/weight/trade-value inputs: holdings participation vs trade-value or weight-delta participation must be distinguished in metric metadata/status.
- [ ] Add regime/scenario diagnostics from existing project-local regime, benchmark, market segment, neutralization, and control columns when present; otherwise emit unavailable/warning rows. Ex-post or missing timing metadata must set `decision_usable=false`; do not treat these rows as point-in-time decision evidence.
- [ ] Add A-share assumption diagnostics rows for limit-up/down, suspension, T+1, short-sale non-executability, baseline cost assumptions, and capacity/tradability warnings. Record but do not simulate these assumptions.
- [ ] Add missing-input tests for `emit_unavailable`, `emit_partial`, and `skip_with_summary`.
- [ ] Add tests that benchmark-relative cards/report sections are hidden or unavailable when benchmark inputs are missing; remaining diagnostics must be worded as descriptive diagnostics, not alpha proof.
- [ ] Keep capacity disabled by default; if explicitly enabled without ADV/liquidity inputs, emit unavailable/warning rows and keep the run valid.
- [ ] Keep `DiagnosticInputBundle` column-pruned in every sub-slice. Diagnostics calculations must not read full project-internal tables or materialize `diagnostic_observations.parquet` unless the explicit detail flag is enabled.
- [ ] Keep data tests at or below current generated-run scale for rows, combinations, date span, and output footprint.

**V1 Metric Family Matrix:**

| Sub-slice | Required inputs | Optional inputs | Missing behavior | Guardrails | Fixture coverage |
| --- | --- | --- | --- | --- | --- |
| 9A common emitter | `combination_id`, policy metadata, base evaluation status | none | fail only on schema-required fields | registry owns field list, grain, unique key, sentinels | disabled, unavailable, partial, skipped rows |
| 9B risk/path | `portfolio/returns.csv` net return series | benchmark return, volatility proxy | unavailable/warning when return series absent | post-run only; does not change returns or weights | volatility, drawdown, tail/ES proxy positive and missing-input cases |
| 9C cost | modeled cost fields or explicit cost component outputs | tax, commission, transfer, slippage, impact assumptions | partial/unavailable; never fill absent components with zero | provenance names modeled vs unmodeled components | component-present, component-missing, break-even/sensitivity rows |
| 9D capacity | weights or trade values plus ADV/liquidity proxy when present | turnover, amount, market cap, float market cap | unavailable/warning when ADV/liquidity absent | distinguish holdings participation from trade-value or weight-delta participation; `capacity_claim=false` for proxy rows | ADV present/absent, holdings vs trade-value metadata |
| 9E regime/scenario/control | base returns and policy metadata | benchmark, market segment, regime, neutralization/control columns | unavailable/warning; ex-post rows set `decision_usable=false` | no future-information leakage; no stress simulation or optimizer | regime present/absent, ex-post scenario, deferred stress absence |
| 9F A-share assumptions | sanitized `config.a_share_assumptions` | tradability flags, industry/neutralization metadata | partial/unavailable; missing tradability fields never mean all tradable | record assumptions only; do not simulate T+1, shorting, order routing, or neutralized portfolio returns | limit/suspension assumptions, missing tradability fields, neutralization metadata-only |

**Tests:**

- `conda run -n analysis python -m pytest -q tests/evaluations/diagnostics tests/evaluations/test_evaluation_engine_contract.py -k diagnostics`

**Acceptance Criteria:**

- Required N5 v1 diagnostics metrics are computed or marked unavailable by policy.
- Required N5 v1 diagnostics metric names, formulas, units, timing requirements, and status behaviors come from `DiagnosticMetricSpec`; implementation-local metric invention fails tests.
- No diagnostics calculation changes base evaluation outputs.
- Family-specific required fields are enforced through registry tests.
- A-share assumptions are recorded and warned about without simulation, paper trading, live trading, or execution-path changes.
- Data-scale acceptance does not exceed current generated-run scale.

**Risks:**

- Diagnostics semantics can expand too far. Keep v1 bounded to post-run explanations, warnings, scenario rows, and proxy metrics; optimizer, portfolio-policy, simulation, execution, capacity curve, and crowding proxy remain out of scope.

## Slice 10: Phase 1B Conditional Materialized Views By Family

**Goal:** Complete N5 v1 conditional diagnostics views one family at a time, each with full registry/display/API/download/frontend coverage before moving to the next.

**Recommended Order:**

1. `dimensions/portfolio_assumptions.csv`
2. `portfolio/weights.parquet`
3. `diagnostics/tradability_diagnostics.csv`
4. `diagnostics/tradability_summary.csv`
5. `diagnostics/scenario_metrics.csv`
6. `diagnostics/portfolio_risk_summary.csv`
7. `diagnostics/exposure_capacity_summary.csv`
8. `diagnostics/diagnostic_observations.parquet` only behind the explicit detail flag and after the first seven are stable.

**Files:**

- Treat `src/quant_research_platform/artifacts/schema_registry.py` as contract-integrator-owned and read-only for view agents; modify it only through a CP5 contract delta if Slice 1 registry entries are incomplete.
- Modify `src/quant_research_platform/evaluations/diagnostics/runner.py`
- Modify `tests/evaluations/diagnostics/`
- Modify `tests/artifacts/test_schema_07_artifacts.py`
- Modify `tests/interfaces/api/test_api.py`
- Modify frontend tests only when a new view changes UI behavior.

**Implementation Steps Per View:**

- [ ] Confirm registry columns, grain, unique key, status fields, preview limits, display eligibility, and download eligibility. If a view needs registry fields missing from Slice 1, stop the view agent and route a CP5 contract-integrator delta through registry, writer/loader, API, frontend, and fixture tests.
- [ ] Add writer payload generation for the view.
- [ ] Add positive loader validation.
- [ ] Add negative validation for missing required columns, invalid nullable/sentinel rules, duplicate unique keys, unsafe paths, and display/download mismatch.
- [ ] Add API preview and direct/ZIP download tests.
- [ ] Add frontend generic display test if the new role or table behavior is not already covered.
- [ ] Add project-internal data tests for each materialized view at or below current generated-run scale; missing optional inputs must produce unavailable/warning output, not fabricated values.
- [ ] Add absence tests proving deferred v1+ artifacts or rows are not accidentally generated: no stress-result rows in `diagnostics/scenario_metrics.csv`, no optimizer/crowding/capacity-curve artifacts, and no `diagnostics/diagnostic_observations.parquet` unless the explicit detail flag is enabled.
- [ ] Stop API/Vite services before running Factor Factory, A-share factor, or evaluation pipeline materialization tests.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/evaluations/diagnostics tests/artifacts tests/interfaces/api`
- `Push-Location web; try { npm test -- --run --maxWorkers=1 } finally { Pop-Location }` when frontend behavior changes.

**Acceptance Criteria:**

- Each conditional view is independently reviewable and can be reverted without breaking required schema `0.7` runs.
- Conditional views never appear as downloadable/display tables without materialized files.
- Scenario metrics have family-specific validators for regime, cost, neutralization, benchmark, and market segment metrics before they become user-visible.
- Materialized-view data tests do not exceed current generated-run scale and are not run concurrently with frontend/backend workbench services.

**Risks:**

- Scenario metrics can become a mixed sparse table. Use registry rules and family validators before exposing it.

## Slice 11A: Current-Contract Docs, Code Maps, And Public Acceptance

**Goal:** Keep durable docs/code maps and public-safe acceptance in the same merge round as schema/API/frontend contract changes. This slice is a Phase 1 merge gate; project-internal data acceptance is handled separately in Slice 11B.

**Files:**

- Modify `ARCHITECTURE.md`
- Modify `EVALUATION_INTERFACE.md`
- Modify `README.md`
- Modify `AGENTS.md`
- Modify `PROJECT_STATUS.md`
- Modify `REPORT.md`
- Modify `RESEARCH_CARD.md`
- Modify `.codex/skills/quant-research-platform-maintenance/SKILL.md`
- Modify `.codex/skills/quant-research-platform-maintenance/references/contracts-and-verification.md`
- Modify relevant `docs/architecture/` entries that still describe schema `0.6` as current.
- Review `docs/architecture/target-architecture-and-capabilities.md`, `docs/architecture/engineering-standards.md`, and `docs/architecture/quant-research-standards.md` for long-term principle conflicts before changing AGENTS/skill/docs wording.
- Modify `docs/code-map/README.md`
- Modify `docs/code-map/evaluation-artifacts-api.md`
- Modify `docs/code-map/platform-flow.md`
- Modify `docs/code-map/platform-flow-interactive.html`
- Modify `notebooks/00_synthetic_ashare_demo.py` and paired notebook/demo artifacts if they read old summary table names.
- Modify `scripts/phase1_acceptance.py`
- Modify `tests/scripts/test_phase1_acceptance.py`
- Create `scripts/check_schema_07_stale_contract.py`
- Create `tests/scripts/test_schema_07_stale_contract.py`
- Create `scripts/check_schema_07_browser_acceptance.ps1`
- Create `web/e2e/schema07-diagnostics.spec.ts`
- Create `scripts/check_schema_07_docs_links.py`
- Create `tests/scripts/test_schema_07_docs_links.py`

**Implementation Steps:**

- [ ] Update docs to say current evaluation-set artifact schema is `0.7`.
- [ ] Treat documentation, `AGENTS.md`, project-local maintenance skill, and code-map updates in this slice as a Phase 1 merge gate for any runtime schema/API/frontend contract change; do not merge Phase 1 with stale schema `0.6` current-contract docs.
- [ ] Update project-level operating rules and project-local maintenance skill references that currently name schema `0.6`, or explicitly mark any retained `0.6` text as historical.
- [ ] Cross-check AGENTS/skill/doc changes against architecture, engineering, and quant research standards. Keep durable principles in `AGENTS.md` or `SKILL.md`; keep current schema/script/route facts in the maintenance skill reference or this implementation plan.
- [ ] Confirm architecture docs stay aligned with this plan: diagnostics live under `evaluations/diagnostics/`; `schema_registry.py` is represented in the target artifact map; N5 v1 risk/cost/capacity/regime proxy diagnostics remain explicitly below full C3 capability completion.
- [ ] Document `run_context.json` replacing separate schema `0.7` `config.json` / `lineage.json`.
- [ ] Document that `run_context.json` is the schema `0.7` lineage carrier and includes the minimum reproducibility fields required by quant standards.
- [ ] Document old diagnostics route removal and generic display table preview route.
- [ ] Document registry-owned field/grain/key/rule source of truth; do not paste full field-count ledgers into Markdown.
- [ ] Add and run a checked stale-current-contract gate instead of relying on review text only. `scripts/check_schema_07_stale_contract.py` must fail on unclassified hits for `schema 0.6`, `artifact_schema_version.*0.6`, `evaluation_id`, `config.json`, `lineage.json`, `ic_summary`, `group_spread_summary`, `performance_summary`, `evaluation_summary`, `summaries/evaluation_summary.csv`, `cumulative_returns`, `portfolio/cumulative_returns.csv`, `label_availability`, `factor_coverage`, `tradability_coverage`, `input_quality/label_availability.csv`, `input_quality/factor_coverage.csv`, `input_quality/tradability_coverage.csv`, `*_summary_by_{year,quarter,month}`, `breakdowns/{period}/{ic_summary,group_spread_summary,performance_summary}.csv`, raw-string selected download item arrays, `/diagnostics/{metric_name}`, `/metrics/{metric_name}`, `/portfolio/{metric_name}`, and `/breakdowns/{group}/{metric_name}`. The script output must classify every remaining hit by `file:line`, class, and reason. Historical docs, negative/rejection tests, `docs/superpowers/specs/`, implementation plans, and intentionally separate Factor Factory schema `0.1` / `factor_factory_run` contracts may be allowlisted only with explicit per-hit reasons. Current-contract sections in `README.md`, `EVALUATION_INTERFACE.md`, `PROJECT_STATUS.md`, `AGENTS.md`, `.codex/skills/`, and `docs/code-map/*` cannot be blanket-allowlisted.
- [ ] Migrate `scripts/phase1_acceptance.py` and `tests/scripts/test_phase1_acceptance.py` to schema `0.7` registry-backed tables such as `metrics/evaluation_metrics_summary.csv`.
- [ ] Update runnable notebooks/demo examples such as `notebooks/00_synthetic_ashare_demo.py` from `ic_summary` / `performance_summary` assumptions to schema `0.7` registry tables.
- [ ] Make `scripts/check_schema_07_browser_acceptance.ps1` plus `web/e2e/schema07-diagnostics.spec.ts` the named checked browser acceptance artifact. The wrapper sequence is fail-closed: verify API/Vite are inactive, load the Slice 8B baseline, verify the required generated run set exists, fail with missing-evidence output if it does not, then start FastAPI and Vite against configured project-local generated run roots, record PIDs/ports/health checks/log paths, run Playwright, and stop only recorded PIDs.
- [ ] Use generated Factor Factory runs and ordinary schema `0.7` evaluation runs for browser acceptance. Required coverage is one Factor Factory listing/detail path, one ordinary evaluation evidence path with diagnostics display, report-null behavior when an ordinary run lacks display-approved `report.md`, display-approved `{kind, key}` download initiation, run switching, and combination-aware refresh where the generated run supports it.
- [ ] Schema `0.7` browser/project acceptance wrappers must load `scripts/private_run_roots.ps1` by default, or pass equivalent explicit `RunRoots` / `PipelineOutputRoot` / `FactorFactoryRoot` values, so Factor Factory acceptance uses the configured project-local Factor Factory root instead of an app fallback root.
- [ ] Slice 11A browser acceptance only consumes existing generated runs. If the required generated run set is missing, do not start API/Vite and do not generate runs inside 11A; report missing evidence and run Slice 11B/project-internal acceptance or the relevant factor-flow generation first, at or below the Slice 8B data-scale baseline. Do not introduce a separate public synthetic fixture generator.
- [ ] Add `scripts/check_schema_07_docs_links.py` and pytest coverage to fail on broken relative links or current-contract file paths that no longer exist in root docs, `docs/`, code maps, and project-local maintenance skill docs.
- [ ] Record open handoff in `PROJECT_STATUS.md` if current topic, phase, milestone, blocker, next step, or data/API/Git/Conda/privacy boundary changes.
- [ ] Review Factor Factory service/API/frontend behavior against schema `0.7` evaluation artifacts; record whether code changes were needed and run targeted Factor Factory tests.

**Tests:**

- `conda run -n analysis python -m pytest -q tests/artifacts tests/interfaces/api`
- `conda run -n analysis python -m pytest -q tests/evaluations tests/markets/implementations/a_share/evaluation`
- `conda run -n analysis python -m ruff check src tests scripts`
- `Push-Location web; try { npm test -- --run --maxWorkers=1; npm run lint; npm run build } finally { Pop-Location }`
- `git diff --check`
- `scripts/verify_code_map_flow.ps1`
- `conda run -n analysis python -m pytest -q tests/scripts/test_schema_07_stale_contract.py`
- Static stale-contract proof gate: `conda run -n analysis python scripts/check_schema_07_stale_contract.py`
- `conda run -n analysis python -m pytest -q tests/scripts/test_schema_07_docs_links.py`
- Docs link/path proof gate: `conda run -n analysis python scripts/check_schema_07_docs_links.py`
- Classified boundary-creep check: `rg -n "vendor API|broker|exchange API|API key|api key|paper trading|live trading|order routing|deployment|production automation" src web/src docs README.md ARCHITECTURE.md EVALUATION_INTERFACE.md REPORT.md RESEARCH_CARD.md PROJECT_STATUS.md AGENTS.md .codex/skills`, with every remaining hit classified as historical/spec/boundary text or removed.

Project-local browser acceptance after frontend/API behavior changes:

- Use generated Factor Factory runs and ordinary schema `0.7` evaluation runs from configured project-local run roots. Commit the browser test/wrapper code, not generated run artifacts, unless the user approves exact tracked fixture paths.
- Start API and Vite through `.\scripts\check_schema_07_browser_acceptance.ps1` or the existing local workbench start helpers it orchestrates only after the wrapper has verified the required generated run set exists, with recorded service PIDs, ports, health checks, and log paths before running browser assertions.
- After Playwright completes, stop the services with `.\scripts\stop_workbench.ps1` when applicable or exact PID cleanup, and list generated Playwright artifacts before any cleanup.
- Run `.\scripts\check_schema_07_browser_acceptance.ps1`, which calls `.\scripts\playwright_cli_project.ps1` against `web/e2e/schema07-diagnostics.spec.ts` to load the workbench, assert generated Factor Factory and ordinary evaluation runs are visible, generic diagnostics tables render from `manifest.display`, preview uses `/tables/{table_key}/preview`, run identity remains scoped by `(source_alias, run_id)`, combination-aware refresh works where supported, optional `report.md` absence shows the intended empty state, selected ZIP initiation starts from display-approved `{kind, key}` items, and one narrow viewport has no obvious overlap.
- Do not read raw/interim private inputs during browser acceptance, expose private paths or row-level data through logs/API/download metadata, or count API-offline `Failed to fetch` states as acceptance evidence.

**Acceptance Criteria:**

- Docs describe implemented behavior only.
- Stale-contract proof gate fails on unclassified schema `0.6` table/path/route identities and passes only with explicit per-hit historical/rejection classifications; current-contract docs and code maps have no blanket stale-contract allowlist.
- Browser acceptance can be reproduced from project-local generated Factor Factory and ordinary evaluation run roots plus recorded service startup evidence.
- Docs link/path proof gate passes after schema `0.7` docs and code-map edits.
- No docs imply deployment, external APIs, paper/live trading, or real trading automation.
- Static source/doc checks have no new trading, deployment, external API, API-key, order-routing, or production-automation creep outside explicitly reviewed historical/spec context.
- No private paths or generated private outputs are tracked.

**Risks:**

- Docs can drift into a second source of truth. Keep docs explanatory and point to registry/tests for exact fields and rules.

## Slice 11B: Project-Internal Data Acceptance Closeout

**Goal:** Run project-internal data acceptance after Phase 1A/1B code/docs verification passes, using existing generated-run scale or smaller and without project-external data/API access.

**Files:**

- Add `scripts/check_schema_07_project_acceptance.ps1`.
- Add `scripts/check_schema_07_project_acceptance.py` or focused helper modules only if the PowerShell wrapper needs structured validation beyond orchestration.
- Add `tests/scripts/test_schema_07_project_acceptance.py` if structured helper logic is introduced.
- Modify `PROJECT_STATUS.md` only if project-internal data acceptance changes current status, blocker, next step, or data/privacy boundary.

**Implementation Steps:**

- [ ] Use only project-internal configured roots and outputs.
- [ ] Default input whitelist is configured run roots plus processed factor/evaluation inputs needed by the A-share evaluation path. Raw/interim CSMAR layers and private reports are read-only and require explicit wrapper flags plus path/count reporting if they are inspected; they must not be regenerated or rewritten in 11B.
- [ ] Use project-internal private data only inside this repository and only for local acceptance; do not print row-level private data, copy outputs outside the project, expose private input roots through API/download metadata, or commit generated private outputs.
- [ ] Keep rows, combinations, date span, output footprint, download/ZIP footprint, frontend preview footprint, and runtime/memory proxy where available at or below the current generated-run scale.
- [ ] Stop API/Vite services before starting Factor Factory, A-share factor, or evaluation pipeline acceptance.
- [ ] Do not access project-external paths, external APIs, credentials, or secrets.
- [ ] Report exact generated paths/counts without printing private data or committing private outputs.
- [ ] Add `scripts/check_schema_07_project_acceptance.ps1` as the single checked 11B entry point. It must verify no recorded API/Vite services are active and also check configured/default/known alternate local API/Vite ports, project workbench lockfiles, and command-line process records. If a matching service cannot be confidently attributed, fail with the exact observed port/process evidence and ask the operator to stop it; do not auto-kill unknown PIDs. The wrapper must run or load the Slice 8B scale baseline from `reports/private_real/schema_07_acceptance/data_scale_baseline.json`, invoke the approved A-share private evaluation path (`python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation`) rather than generic `evaluate-parquet` for A-share acceptance, run Factor Factory/factor-summary smoke checks when those paths were migrated, run manifest/display/download/private-path leak checks, compare output scale against baseline, and print only project-relative generated paths plus aggregate counts.

**Acceptance Matrix:**

| Surface | Required evidence | Boundary |
| --- | --- | --- |
| A-share evaluation path | one existing-scale-or-smaller configured run that writes schema `0.7` diagnostics files | project-internal paths only; no row-level private data in logs |
| Factor Factory / factor-summary, if migrated | targeted run or fixture proving schema `0.7` summary consumption and stable `factor_summary.csv` output | no retirement of `factor_factory_run` contract |
| Generic Parquet/uploaded-panel, if used | small configured local input or generated run proving disabled diagnostics path | no project-external data/API |
| Artifacts and downloads | path/count summary, manifest/display validation, no private path leakage | private/generated outputs remain ignored/untracked |
| Resource sequencing | recorded absence of API/Vite services during factor/evaluation data runs | no concurrent browser acceptance and factor-flow runs |

**Tests:**

- `.\scripts\check_schema_07_project_acceptance.ps1`
- `conda run -n analysis python -m pytest -q tests/scripts/test_schema_07_project_acceptance.py` if helper logic is introduced.

**Acceptance Criteria:**

- Project-internal data acceptance uses existing-scale-or-smaller evidence and reports exact paths/counts without leaking private data.
- A-share acceptance uses the A-share private evaluation entry point, not generic `evaluate-parquet`.
- The checked wrapper records service absence, baseline comparison, generated paths/counts, manifest/display/download validation, and private-path leak checks in one reproducible command.

**Risks:**

- Resource contention can distort acceptance. Do not run frontend/backend services concurrently with Factor Factory or evaluation pipeline data runs.

## Phase Grouping

Phase 1A, schema/API/display contract shell:

- Slice 1: Schema registry source of truth.
- Slice 2A: Diagnostics disabled/spec/policy shell.
- Slice 2B: Schema `0.7` writer/loader synthetic artifact.
- Slice 2C + Slice 2D: Runtime switch, lightweight index, payload/comparison migration, active time-breakdown decision, and Factor Factory/factor-summary active-consumer pre-switch gate. These are one review gate even if implemented in separate commits.
- Slice 3: Diagnostics input bundle and enabled policy runner.
- Slice 4: Registry-driven `manifest.display`.
- Slice 5: API preview/download and legacy route removal.
- Slice 6: Minimal generic frontend diagnostics display.
- Slice 7A: Backend diagnostics request contract for snapshots/runs, disabled/no-view only.
- Slice 8: Conditional view requestability gate, limited to policy/request wiring for Slice 1 registered path/key/status/display eligibility.
- Slice 7B: A-share pipeline, generic Parquet adapter, CLI/config, and diagnostics options discovery after Slice 8.
- Slice 7C: Factor Factory inspection/migration and factor-summary schema `0.7` migration.
- Slice 11A current-contract docs/code-map/AGENTS/maintenance-skill updates and public-safe acceptance, after Phase 1A/1B tests pass but before merge.

Phase 1B, complete N5 v1 diagnostics depth:

- Slice 8B: Data scale and resource preflight before materialized diagnostics data tests.
- Slice 9: N5 v1 materialized diagnostics calculations.
- Slice 10: N5 v1 conditional materialized views by family.
- Slice 7D: Pipeline Builder diagnostics-options migration after Slice 10/CP5 shared `DiagnosticsOptions` refresh. Uploaded-panel disabled UX is already complete through `7D0` in Slice 6.

Phase 1 public-safe merge readiness requires Phase 1A, Phase 1B, and Slice 11A docs/code-map/public acceptance. Because project-internal data acceptance is required for this implementation goal, full schema `0.7` / N5 v1 completion also requires Slice 11B. Phase 1A alone is a checkpoint, not the final goal.

Phase 1C, project-internal data acceptance closeout:

- Slice 11B project-internal data acceptance only, after Phase 1A/1B code/docs verification, using existing generated-run scale or smaller.

## Implementer Self-Checks

Before starting implementation, perform these self-checks without asking the user again. Do not ask for user input unless implementation would cross an approval boundary listed in `AGENTS.md` or this plan.

- [ ] Schema `0.7` replaces schema `0.6` with no compatibility wrapper.
- [ ] Implementers have read the locked run identity decision: schema `0.7` portable artifacts use `run_id` only and must not serialize `evaluation_id`; API routes, API payloads, downloads, cache keys, and frontend route/cache state still preserve runtime identity as `(source_alias, run_id)`.
- [ ] Default optional inputs are benchmark return, industry, market cap, volatility, liquidity, and beta proxy only when already available locally.
- [ ] `diagnostics/diagnostic_observations.parquet` remains disabled by default.
- [ ] Project-internal data may be used for implementation and acceptance, while private data must not be printed, copied outside the project, committed, or exposed through API/download metadata.

Before merging Phase 1:

- [ ] Registry tests prove field counts, table counts, grains, unique keys, required-if, nullable-if, sentinel rules, and conditional table eligibility.
- [ ] Registry tests prove stable `sort_key` metadata and `default_materialized=false` for every generated/conditional table.
- [ ] Shared `ApiErrorEnvelope`, `UploadedPanelCreateRequest` / `UploadedPanelCreateResponse`, `DiagnosticsOptions`, display/download refs, and TypeScript API types are frozen before their corresponding dependent API/frontend/pipeline slices.
- [ ] Writer/loader/index reject schema `0.6`.
- [ ] Required files and required tables are enforced by both full loader and lightweight index.
- [ ] `manifest.display` validates every exposed table/download/preset/chart.
- [ ] Registry/display tests cover schema `0.7` time-breakdown generated tables and the API-derived primary cumulative chart spec.
- [ ] API preview, download, ZIP, and comparison paths use display/schema `0.7` metadata only.
- [ ] Direct downloads and selected ZIP use the same `{kind, key}` display refs; raw path direct downloads are rejected or historical-only for schema `0.7`.
- [ ] Portable `manifest.display` contains only artifact-backed items; runtime comparison downloads are API augmentations and cannot weaken manifest validation.
- [ ] Retired schema `0.6` artifact identities, including `group_spread_summary`, `evaluation_summary`, and `portfolio/cumulative_returns.csv`, are migrated/replaced with tests and cannot appear in active schema `0.7` manifests.
- [ ] Frontend renders diagnostics generically from display.
- [ ] Slices 5 and 6 pass as one integration checkpoint: backend legacy route removal and frontend generic diagnostics/download consumers are updated together.
- [ ] Legacy diagnostics route is removed.
- [ ] Legacy path-inference preview routes are removed or explicitly rejected for schema `0.7`.
- [ ] Pipeline, CLI, backend direct-run, and Pipeline Builder entry points all construct the same `DiagnosticsSpec`; old `PipelineConfig.diagnostics: tuple[str, ...]` behavior is gone.
- [ ] `evaluate-parquet` and `datasets/parquet_panel.py` write schema `0.7` diagnostics-disabled artifacts, not schema `0.6` outputs.
- [ ] Factor Factory consumers have been reviewed and migrated against schema `0.7` evaluation artifacts, or proven inactive/non-applicable with targeted grep/test evidence.
- [ ] The read-only data-scale preflight records current generated-run maxima before Phase 1B/project-internal data tests.
- [ ] Phase 1B N5 v1 diagnostics metrics and conditional materialized views pass registry, writer/loader, API/download, frontend, and project-internal data tests.
- [ ] Project-internal data tests do not exceed current generated-run scale and are not run concurrently with API/Vite services.
- [ ] Slice 11B passes through the checked project-internal acceptance wrapper.
- [ ] Current-contract docs, code maps, `AGENTS.md`, and project-local maintenance skill references are updated in the same Phase 1 merge round.
- [ ] Checked stale-contract script and test pass with no unclassified current-contract schema `0.6` hits.
- [ ] No private paths, `source_alias`, credentials, API keys, or machine-specific paths are serialized into portable artifacts.

## Verification Profile

Use narrow tests first:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts/test_schema_07_registry.py
conda run -n analysis python -m pytest -q tests/artifacts/test_schema_07_artifacts.py
conda run -n analysis python -m pytest -q tests/evaluations/diagnostics
conda run -n analysis python -m pytest -q tests/interfaces/api/test_api.py
Push-Location web; try { npm test -- --run --maxWorkers=1 } finally { Pop-Location }
```

Then broad contract checks when the touched surface justifies it:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts tests/interfaces/api
conda run -n analysis python -m pytest -q tests/evaluations tests/markets/implementations/a_share/evaluation tests/pipelines/a_share tests/research/factor_factory tests/interfaces/cli tests/scripts
conda run -n analysis python -m pytest -q tests/scripts/test_phase1_acceptance.py
conda run -n analysis python scripts/check_schema_07_stale_contract.py
conda run -n analysis python scripts/check_schema_07_docs_links.py
conda run -n analysis python -m ruff check src tests scripts
Push-Location web; try { npm test -- --run --maxWorkers=1; npm run lint; npm run build } finally { Pop-Location }
.\scripts\verify_code_map_flow.ps1
git diff --check
git status --short
```

Browser acceptance is required only after frontend behavior changes and must start both FastAPI and Vite. Do not count an API-offline `Failed to fetch` state as acceptance evidence for diagnostics.

Required acceptance wrappers for this complete schema `0.7` / N5 v1 goal:

```powershell
.\scripts\check_schema_07_browser_acceptance.ps1
.\scripts\check_schema_07_project_acceptance.ps1
```

For this complete schema `0.7` / N5 v1 goal, `.\scripts\check_schema_07_project_acceptance.ps1` is mandatory after broad verification. `.\scripts\check_schema_07_browser_acceptance.ps1` is mandatory because this plan changes API/frontend evidence behavior, unless implementation records that no API/frontend behavior changed.

After wrappers and `git status --short`, classify every modified or untracked generated output before final review, especially browser acceptance outputs, Playwright screenshots/traces/logs, ZIP caches, private acceptance outputs, and generated reports. Keep ignored/generated evidence out of commits unless the user approves exact tracked fixture paths.
If generated Playwright/log/ZIP/cache artifacts need cleanup, leave them in place and report exact paths unless exact cleanup approval is already present.

Final review gate: after broad checks and required wrappers pass, dispatch a cold reviewer against the final diff. Close the implementation only when the final review reports no confirmed P0/P1/P2 findings. If review finds issues, fix them, rerun the narrowest affected checks plus any required wrapper touched by the fix, and run a focused cold re-review.

Resource sequencing:

- Do not run API/Vite browser acceptance concurrently with Factor Factory, A-share factor, generic Parquet, or evaluation pipeline data runs.
- Before starting project-internal data acceptance or factor-flow tests, verify recorded service PIDs, configured/default/known alternate local API/Vite ports, project workbench lockfiles, and command-line process records are inactive. Fail closed if service ownership is uncertain; do not auto-kill unknown PIDs.
- Keep project-internal data tests at or below the current generated-run scale across rows, combinations, date span, output footprint, download/ZIP footprint, frontend preview footprint, and runtime/memory proxy where available unless the user separately approves a larger run.

## Execution Recommendation

Use subagent-driven development for implementation:

- Contract integrator: exclusively owns `schema_registry.py`, registry-owned `ChartSpec` / `DiagnosticMetricSpec` / `DiagnosticOptionalInputSpec` / `manifest.artifacts` adapters, `artifact_package.py`, `display_manifest.py`, `display_tables.py`, shared schema/display/download wire shape, `interfaces/api/models.py`, `interfaces/api/payloads.py`, `web/src/shared/api/api.ts`, `web/src/shared/downloads.ts`, canonical shared fixture builders (`tests/artifacts/artifact_fixtures.py`, `tests/interfaces/api/artifact_run_fixtures.py`, `web/src/shared/testFixtures.ts`), and any cross-layer type changes. The evaluation diagnostics owner owns `src/quant_research_platform/evaluations/diagnostics/` implementation files after the integrator lands registry-facing contracts. Other agents treat these files as read-only until the owning sub-slice contract lands.
- Agent A: Slice 2A/2B diagnostics shell and writer/loader synthetic artifact tests, working through contract-integrator-owned registry/type checkpoints rather than editing shared registry/type files directly.
- Agent B: Slice 2C/5 backend run index, runtime switch, time-breakdown migration, API preview/download/route removal, upload-run migration, and comparison tests after the shared contract lands.
- Agent C: Slice 6 frontend generic display, `7D0` uploaded-panel disabled UX, shell tests, download picker, `web/src/features/evidence/useArtifactDownloads.ts` UI state, API-derived chart rendering, report-null state, and TypeScript tests after shared API/display/upload types land.
- Agent C2: Slice 7D Pipeline Builder diagnostics-options UI after Slice 10/CP5 refreshes `DiagnosticsOptions`; owns `PipelineBuilder.tsx` and pipeline frontend tests only. Uploaded-panel create/delete UX is not part of C2 because it is already completed in Slice 6/`7D0`.
- Agent D: Slice 2D plus Slice 7A/7B/7C request surfaces, A-share pipeline, generic Parquet adapter, CLI/config, Factor Factory, and factor-summary migration. Agent D is the single owner for Factor Factory/factor-summary migration: 2D completes all active read-path migration needed before the runtime switch, while 7C is limited to request-surface deltas or static proof that no active schema `0.6` read path remains. Respect the 7A -> Slice 8 -> 7B -> 7C order. Slice 7D is deferred until after Slice 10/CP5 refreshes implemented/requestable diagnostics options.
- Main agent: serialize shared-contract edits, integrate subagent outputs, enforce no compatibility wrapper, run verification, and update docs only after code passes.

Contract checkpoints:

`CP*` labels below are implementation coordination gates. They are not the roadmap capability labels such as C0-C8, and `CP3` must not be confused with roadmap C3 risk/regime/cost/capacity capability.

- CP0 before Slice 1 consumers: contract integrator lands registry table/file specs and the schema `0.6` artifact identity decision table. Other agents may add tests against the registry but do not redefine field lists in their slices.
- CP1a before Slice 2A consumers: contract integrator lands the minimal shared Python/TypeScript `ApiErrorEnvelope`, diagnostics disabled request shape, and fixture builders. Slice 2A consumes those shapes and tests diagnostics semantics; it does not invent local error/request objects.
- CP1 before Slices 2B/2C consumers: contract integrator lands required metadata/table registry, time-breakdown generated specs, `ChartSpec`, `manifest.artifacts` adapter shape, and synthetic artifact fixture builders. Other agents consume these shapes read-only.
- CP2 before Slices 4/5/6 consumers: contract integrator lands `manifest.display` table/download/chart item identity, selected-download `{kind, key}` shape, and shared TS API fixtures. Backend and frontend agents do not invent alternate download ids.
- CP3 before Slice 5 route migration: contract integrator lands API payload/model shapes for evidence, preview, direct download, ZIP download, uploaded-panel create/delete, and comparison. Route agents may request deltas, but changes go back through the integrator.
- CP4 before Slice 7B options work: contract integrator lands initial `DiagnosticsOptions` with registered/requestable/implemented semantics after Slice 8. Pipeline/backend agents consume that shared type only.
- CP5 before Slice 7D and any post-Slice-10 UI/options work: contract integrator refreshes `DiagnosticsOptions`, schema registry deltas if any, shared fixtures, API payloads, and TypeScript types after materialized views become implemented/requestable. View agents must not edit `schema_registry.py` directly; missing registry fields discovered in Slice 10 stop at this checkpoint.
- Fixture shape changes after CP1/CP2/CP3 require an integrator delta before dependent agents continue. Subagents may add slice-local fixtures inside their own test files, but schema `0.7` canonical artifact/API/frontend fixture builders remain contract-integrator owned.

Keep write scopes disjoint per slice. Do not let multiple agents edit the same file concurrently. If a slice discovers a needed change in contract-owned files, stop that slice at the checkpoint and route the delta through the contract integrator before continuing. Slices 5 and 6 are a paired integration checkpoint: they may be reviewed separately, but route removal/download wire changes are not mergeable until frontend consumers and shell tests are migrated in the same branch.

## Implementation Handoff

This plan is confirmed for implementation. Do not run another broad plan/spec cold review before starting unless new user requirements arrive or implementation discovers a new P0/P1/P2 blocker not covered here.

Locked implementation defaults:

- Implement diagnostics code under `src/quant_research_platform/evaluations/diagnostics/`; do not create a top-level `src/quant_research_platform/diagnostics/` package.
- Treat `src/quant_research_platform/artifacts/schema_registry.py` and registry-backed tests as the source of truth for field counts, grains, unique keys, conditional rules, sentinel rules, required generated files, and display/download eligibility.
- Schema `0.7` fully replaces schema `0.6`; do not add compatibility wrappers, fallback loaders, or schema `0.6` UI/API compatibility.
- N5 v1 risk/regime/cost/capacity outputs are descriptive proxy diagnostics only; they do not complete roadmap C3 and must not be worded as alpha proof, real capacity, validated risk model, paper trading, live trading, or investment advice.
- Use project-local generated Factor Factory runs and ordinary schema `0.7` evaluation runs for browser acceptance. Wrappers must load `scripts/private_run_roots.ps1` by default or pass equivalent explicit roots.
- Project-internal data may be read for implementation and acceptance within this plan's limits; private rows/paths must not be printed, copied outside the project, committed, or exposed through portable artifacts/API/download metadata.
- Keep project-internal data tests at or below current generated-run scale. Generate the ignored data-scale baseline before Phase 1B/project-internal acceptance and compare later runs against it.

Expected implementation flow:

1. Open a goal for this plan and use subagent-driven development for disjoint slices.
2. Let the contract integrator land registry/shared contract checkpoints before dependent agents edit consumers.
3. Implement slices through code, tests, docs, and wrappers; do not expand the spec.
4. After implementation, run cold review with subagents. Fix confirmed P0/P1/P2 findings, rerun affected checks, and repeat focused cold review until no confirmed P0/P1/P2 remains.

Only stop for the user if implementation would cross one of these boundaries: dependency/browser installation, unknown service ownership, larger-than-baseline data runs, project-external access, external API/secrets, raw/interim/processed private data rewrite, file/directory deletion or move without exact-path approval, or paper/live trading/deployment behavior.
