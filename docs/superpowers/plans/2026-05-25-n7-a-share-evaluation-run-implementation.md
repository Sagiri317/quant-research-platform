# N7 A-share Evaluation Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Implementation status (2026-05-26):** This plan has been executed and merged to `main`. Public metadata sanitization, schema `0.6` metadata key validation, Factor Factory N7 artifact directory handling, and docs drift fixes are part of the current baseline. Treat the detailed checkbox list below as the historical implementation plan, not as a fresh pending task list. Current code/docs/tests are the source of truth for follow-up review.

**Goal:** Implement the finalized N7 A-share evaluation run contract: one user-visible evaluation run contains a matrix of factor/label/kernel/portfolio-mode combinations, writes artifact schema `0.6`, supports low-memory comparison/reporting, and lets the frontend fully inspect the resulting tables and summaries.

**Architecture:** Keep the public language as "evaluation run/results"; use `EvaluationMatrixRun` only as an internal implementation concept. The A-share adapter owns private-data construction and matrix expansion. The generic evaluation/artifact/API/frontend layers consume explicit schema `0.6` artifacts through `manifest.display`; they must not rely on A-share-only shortcuts.

**Tech Stack:** Python, pandas, NumPy, PyArrow/Parquet, FastAPI, React/Vite, Vitest, pytest, ruff, local Conda environment `analysis`.

---

## Original Pre-Implementation Starting Point

- Final design spec: `docs/superpowers/specs/2026-05-25-n7-a-share-run-identity-design.md`.
- At plan creation, artifact schema was `0.5` and still used `positions.csv`, `pnl.csv`, root-level `combinations.csv`, and `engine` language.
- At plan creation, the runner supported `spec.engine` and combination axes `feature`, `label`, `weight_id`, `cost_id`.
- At plan creation, loader and comparison paths read whole CSV artifacts; they did not support Parquet table metadata or low-memory large-run comparison.
- At plan creation, the frontend depended on `manifest.display`, but display metadata did not yet expose table format, dimension source, join key, or preview row intent.
- Target first private smoke run is 2023H1 only. The full 2023-2025 canonical run is a later acceptance run after implementation is verified.

## Target First-Stage Run

- Output root: `reports/private_real/factor_evaluation/n7_eval_2023h1_smoke_v1`.
- Canonical future output root: `reports/private_real/factor_evaluation/n7_eval_2023_2025_v1`.
- Factor selection: first smoke uses explicit feature columns only. N7 first stage does not define extra input shortcuts.
- Features: `mom_20d`, `vol_20d`, `avg_amount_20d`.
- Labels: `forward_return_5d`, `forward_return_20d`.
- Kernels: `tensor`, `dataframe`.
- Portfolio modes: `long_only`, `long_short`.
- Cost config: zero cost, recorded as a run-level config.
- Benchmark config: no benchmark in first stage, recorded as `benchmark_profiles=[]` in run-level config.
- Total combinations: 24.
- `long_short` is a diagnostic spread/grouped portfolio mode, not a real short-selling strategy.

## File Map

### Generic Evaluation

- `src/quant_research_platform/evaluations/evaluation_spec.py`
- `src/quant_research_platform/evaluations/evaluation_runner.py`
- `src/quant_research_platform/evaluations/evaluation_result.py`
- `src/quant_research_platform/kernels/protocol.py`
- `src/quant_research_platform/kernels/dataframe.py`
- `src/quant_research_platform/kernels/tensor.py`
- `tests/evaluations/`
- `tests/kernels/`

### Artifact Schema, Loader, Display, Comparison

- `src/quant_research_platform/artifacts/artifact_schema.py`
- `src/quant_research_platform/artifacts/artifact_writer.py`
- `src/quant_research_platform/artifacts/artifact_loader.py`
- `src/quant_research_platform/artifacts/display_manifest.py`
- `src/quant_research_platform/artifacts/manifest_validation.py`
- `src/quant_research_platform/artifacts/artifact_comparison.py`
- `src/quant_research_platform/artifacts/evaluation_comparison_report.py` (new)
- `tests/artifacts/`

### A-share Evaluation and Pipeline

- `src/quant_research_platform/markets/implementations/a_share/evaluation/evaluation_adapter.py`
- `src/quant_research_platform/markets/implementations/a_share/evaluation/run_factor_evaluation.py`
- `src/quant_research_platform/markets/implementations/a_share/evaluation/matrix_spec.py` (new)
- `src/quant_research_platform/markets/implementations/a_share/evaluation/matrix_runner.py` (new)
- `src/quant_research_platform/markets/implementations/a_share/evaluation/matrix_identity.py` (new)
- `src/quant_research_platform/pipelines/a_share/`
- `tests/markets/implementations/a_share/evaluation/`
- `tests/pipelines/a_share/`

### API and Frontend

- `src/quant_research_platform/interfaces/api/`
- `tests/interfaces/api/`
- `web/src/shared/`
- `web/src/features/`
- `web/src/shell/`
- `web/src/**/*.test.*`

### Factor Factory

- `src/quant_research_platform/research/factor_factory/`
- `tests/research/factor_factory/`

---

## Implementation Tasks

### 1. Lock Schema 0.6 Contract in Tests First

- [ ] Add artifact schema `0.6` contract tests before changing the writer.
- [ ] Encode required files exactly:
  - [ ] `manifest.json`
  - [ ] `summary.json`
  - [ ] `config.json`
  - [ ] `lineage.json`
  - [ ] `report.md`
  - [ ] `dimensions/combinations.csv`
  - [ ] `metrics/ic.csv`
  - [ ] `metrics/ic_summary.csv`
  - [ ] `metrics/grouped_returns.csv`
  - [ ] `metrics/group_spread_summary.csv`
  - [ ] `metrics/performance_summary.csv`
  - [ ] `portfolio/returns.csv`
  - [ ] `portfolio/cumulative_returns.csv`
  - [ ] `portfolio/weights.parquet`
  - [ ] `diagnostics/tradability_diagnostics.csv`
  - [ ] `input_quality/label_availability.csv`
  - [ ] `input_quality/factor_coverage.csv`
  - [ ] `input_quality/tradability_coverage.csv`
  - [ ] `summaries/evaluation_summary.csv`
- [ ] Encode the required combination columns:
  - [ ] `combination_id`
  - [ ] `feature`
  - [ ] `label`
  - [ ] `kernel`
  - [ ] `portfolio_mode`
- [ ] Encode `portfolio/weights.parquet` as required Parquet output with `target_weight`.
- [ ] Encode `portfolio/cumulative_returns.csv` as the cumulative return time-series table; do not keep `pnl.csv` in schema `0.6`.
- [ ] Encode `summaries/evaluation_summary.csv` as the compact run summary table; do not use `matrix_summary.csv`.
- [ ] Make optional/future tables explicit in schema metadata but not required for first-stage validation; do not implement optional/future table writers unless an existing result already provides that table.
- [ ] Treat `dimensions/combinations.csv` as a dynamic dimension table: first stage required columns are only `combination_id`, `feature`, `label`, `kernel`, and `portfolio_mode`.
- [ ] Keep run-level constant cost, weight, benchmark, universe, eligibility, and tradability configuration in `config.json` / `lineage.json`, not in `combinations.csv`.
- [ ] Preserve actual cost, turnover, target weight, and diagnostic method values in their fact tables.
- [ ] Put future `*_profile_id` or `*_policy_id` columns into `combinations.csv` only when the same run evaluates multiple profile/policy values.

### 2. Rename Evaluation Engine Language to Kernel

- [ ] Rename QRP-owned public fields from `engine` to `kernel`.
- [ ] Rename helper symbols where they are QRP-owned, such as `EVALUATION_ENGINES` and `normalize_engine_name`, to kernel names.
- [ ] Keep third-party library keyword arguments named `engine` where pandas, pyarrow, or other libraries require that exact keyword.
- [ ] Remove compatibility with old external `engine` request/config fields.
- [ ] Update CLI/API/pipeline tests to assert old `engine` is rejected or absent from generated config.
- [ ] Update docs and code comments touched by the implementation so they consistently say `kernel` for dataframe/tensor execution.

### 3. Add A-share Matrix Run Identity Layer

- [ ] Add an internal A-share matrix spec that expands one requested evaluation run into explicit combinations.
- [ ] Use explicit `--features` as the only first-stage feature selection input.
- [ ] Do not implement named feature shortcuts or a feature selection registry in N7 first stage.
- [ ] Validate that every requested feature column exists in the input table.
- [ ] Recommended new files:
  - [ ] `matrix_spec.py` for typed request/config objects.
  - [ ] `matrix_identity.py` for stable IDs and dimension defaults.
  - [ ] `matrix_runner.py` for orchestration over kernels and portfolio modes.
- [ ] Keep the API-visible run as one run directory with one `run_id`.
- [ ] Keep all factor/label/kernel/mode variations inside `dimensions/combinations.csv`.
- [ ] Generate deterministic `combination_id` values from the required dimensions.
- [ ] Keep first-stage `benchmark_profiles=[]` in run-level config; future benchmark tables should use `benchmark_profile_id`.
- [ ] Keep `signal_id`, `model_id`, and future axes out of first-stage required columns.

### 4. Refactor Generic Evaluation Output to Schema 0.6

- [ ] Change the generic runner to emit schema `0.6` table names and paths.
- [ ] Replace `positions` output with `weights`.
- [ ] Replace `position` column semantics with `target_weight`.
- [ ] Replace `pnl` output with `cumulative_returns`.
- [ ] Replace `pnl_return_column` config naming with `return_source_column`.
- [ ] Remove old `weight_id` and `cost_id` from combination rows when they are run-level constants.
- [ ] Add only `portfolio_mode` and `kernel` to first-stage combination rows beyond existing `feature` and `label`.
- [ ] Write `lineage.json` with input data references, run config hash, schema version, and generation timestamp.
- [ ] Write `input_quality/*` tables from available validation and coverage diagnostics.
- [ ] Write `summaries/evaluation_summary.csv` from the same aggregate information the UI needs for quick inspection.
- [ ] Preserve narrow unit tests around feature/label evaluation logic while changing only the output contract.

### 5. Add Parquet-Aware Artifact Writer and Loader

- [ ] Update the artifact writer to support mixed table formats per schema metadata.
- [ ] Write `portfolio/weights.parquet` through a structured writer path, not by hardcoded special cases scattered across callers.
- [ ] Update the loader so callers can load selected tables and understand CSV versus Parquet from schema/display metadata.
- [ ] Keep selected-table loading lazy enough that API preview and comparison can avoid reading full `weights.parquet`.
- [ ] Update validation to check Parquet existence and required columns without loading the entire table into memory.
- [ ] Reject schema `0.5` artifacts in schema `0.6` formal validation paths instead of silently adapting them.

### 6. Implement DataFrame Kernel Reference Path

- [ ] Make dataframe execution the first reference implementation for the new schema.
- [ ] Ensure dataframe output includes all 24 target combinations for the 2023H1 smoke configuration.
- [ ] Verify all required files are written for a small synthetic fixture.
- [ ] Verify `long_only` and `long_short` generate distinct `portfolio_mode` rows and summaries.
- [ ] Verify missing factor/label availability appears in `input_quality/*` rather than causing opaque downstream failures.

### 7. Implement Tensor Kernel Parity Path

- [ ] Keep `tensor` as a first-class kernel, not a temporary adapter-only mode.
- [ ] Route tensor execution through the same matrix run and artifact writer contract.
- [ ] Ensure tensor output tables use the same schema, table names, column names, and combination IDs as dataframe output.
- [ ] Add fixture parity tests for dataframe versus tensor on a small deterministic panel.
- [ ] Use tolerance suited to vectorized floating-point computation:
  - [ ] exact equality for schema, table names, row keys, IDs, categorical fields, and non-floating columns.
  - [ ] `rtol=1e-10`, `atol=1e-12` for stable floating aggregates on the synthetic fixture.
  - [ ] a looser documented tolerance may be used only for private large-sample acceptance if numeric backend differences require it.

### 8. Update A-share CLI, Adapter, and Pipeline Integration

- [ ] Update the A-share evaluation adapter to construct the N7 matrix run from processed A-share factor data.
- [ ] Update `run_factor_evaluation.py` to accept `--kernels` and the other multi-value matrix options.
- [ ] Remove old `--engine` public usage from A-share CLI examples and tests.
- [ ] Keep date filtering behavior explicit:
  - [ ] `--start-year/--end-year` defines an evaluation window by calendar year.
  - [ ] `--start-date/--end-date` defines an exact evaluation window and can be used instead of year arguments.
  - [ ] If year arguments are present and exact dates are absent, evaluate from January 1 of `start-year` through December 31 of `end-year`.
  - [ ] If both year and exact-date arguments are absent, evaluate the full available input table.
  - [ ] If both year and exact-date arguments are present, exact dates must be inside the year range; otherwise fail fast with a clear validation error.
  - [ ] `--input-file` can bypass automatic processed-factor-file resolution for explicitly named private Parquet files in backend/CLI usage only.
  - [ ] Without `--input-file`, the A-share adapter resolves a processed factor file that covers the requested evaluation window.
- [ ] Add a dry-run or plan-printing path that shows expanded combinations without writing private artifacts.
- [ ] Implement `--features`, `--labels`, `--kernels`, and `--portfolio-modes` as one flag followed by multiple values, not repeated flags.
- [ ] Update A-share pipeline run integration so executable pipeline runs write schema `0.6` evaluation artifacts.
- [ ] Keep `--input-file` out of API request models, frontend forms, pipeline builder payloads, and saved pipeline snapshots.
- [ ] Preserve the current endpoint split:
  - [ ] `POST /api/pipeline-snapshots` saves configuration only.
  - [ ] `POST /api/pipeline-runs` executes through evaluation with `start_from`.

### 9. Update Manifest Display and Evidence API

- [ ] Extend `manifest.display` entries with enough metadata for frontend and API behavior:
  - [ ] table key.
  - [ ] label.
  - [ ] relative path.
  - [ ] format: `csv`, `parquet`, `json`, or `markdown`.
  - [ ] role: dimensions, metrics, portfolio, diagnostics, input_quality, summary, report.
  - [ ] optional `dimension_source`.
  - [ ] optional `join_key`.
  - [ ] preview row limit intent, defaulting to 20 for large tables.
- [ ] Update evidence routes to serve schema `0.6` display metadata without backend-only shortcuts.
- [ ] Add API tests proving `--input-file` is not accepted or exposed through frontend/API-facing pipeline and evaluation request payloads.
- [ ] Update table preview routes to preview Parquet safely with a 20-row default.
- [ ] Keep ZIP and direct-download behavior in parity with `manifest.display`.
- [ ] Preserve API-visible run identity `(source_alias, run_id)` in list, detail, preview, direct download, ZIP, and comparison routes.

### 10. Update Frontend Evidence Workbench

- [ ] Update shared API types to understand schema `0.6` display metadata and Parquet table entries.
- [ ] Update run detail views to show the new required tables.
- [ ] Add selectors or tabs that let the user inspect results by label, kernel, portfolio mode, and feature/combination where data is available.
- [ ] Keep frontend preview to 20 rows for large artifact tables, including `portfolio/weights.parquet`.
- [ ] Keep UI practical and complete; do not spend this task on visual redesign.
- [ ] Update frontend tests for run switching, combination switching, table preview, direct download, ZIP picker parity, and comparison entry points.

### 11. Refactor Comparison Surfaces

- [ ] Refactor `compare_evaluation_sets` into a low-memory web/API summary comparison.
- [ ] Ensure it reads only compact metadata and summary tables by default:
  - [ ] `manifest.json`
  - [ ] `summary.json`
  - [ ] `dimensions/combinations.csv`
  - [ ] `metrics/ic_summary.csv`
  - [ ] `metrics/performance_summary.csv`
  - [ ] `summaries/evaluation_summary.csv`
- [ ] Do not read `portfolio/weights.parquet` for web/API summary comparison.
- [ ] Add `evaluation_comparison_report.py` for file-only detailed comparison reports.
- [ ] Make the detailed report compare schema `0.6` combinations, kernel parity, and selected metric/time-series differences.
- [ ] Output detailed comparison as CSV, JSON, and Markdown.
- [ ] Do not make the detailed report part of the main web UI in first stage.
- [ ] Do not read or adapt schema `0.5` artifacts in formal comparison code.

### 12. Adapt Factor Factory to N7

- [ ] Inspect current Factor Factory persistent artifact shape before changing it.
- [ ] Update Factor Factory to call or reference the N7 evaluation run contract instead of forcing N7 to match old Factor Factory summaries.
- [ ] Keep each feature independently evaluable as a combination while still allowing the run to calculate all features together.
- [ ] If Factor Factory stores only `run_ref` and compact candidate summaries, keep its schema unchanged.
- [ ] If Factor Factory persists per-combination evaluation details, bump its schema and add migration-free tests for the new format.
- [ ] Ensure Factor Factory UI/API links can navigate to the single evaluation run and then select the relevant combination.

### 13. Documentation Updates

- [ ] Update `EVALUATION_INTERFACE.md` for schema `0.6`, kernel naming, matrix-run semantics, and required artifacts.
- [ ] Update `ARCHITECTURE.md` only where module boundaries changed.
- [ ] Update affected files under `docs/code-map/`, especially platform flow, A-share data/pipeline, and evaluation/artifact/API/frontend contract maps.
- [ ] Update `PROJECT_STATUS.md` with the current open N7 implementation signal after code changes land.
- [ ] Update `docs/architecture/next-stage-roadmap.md` only if task priority/status changes.
- [ ] Update `.codex/skills/quant-research-platform-maintenance/SKILL.md` only if the reusable maintenance workflow changes.
- [ ] Keep private run results and one-time acceptance evidence out of durable global rules.

### 14. Verification

- [ ] Run narrow Python tests first:
  ```powershell
  $env:PYTHONPATH="src"
  conda run -n analysis python -m pytest -q tests/kernels tests/evaluations tests/artifacts
  ```
- [ ] Run A-share and pipeline tests:
  ```powershell
  $env:PYTHONPATH="src"
  conda run -n analysis python -m pytest -q tests/markets/implementations/a_share/evaluation tests/pipelines/a_share
  ```
- [ ] Run API and Factor Factory tests:
  ```powershell
  $env:PYTHONPATH="src"
  conda run -n analysis python -m pytest -q tests/interfaces/api tests/research/factor_factory
  ```
- [ ] Run lint:
  ```powershell
  $env:PYTHONPATH="src"
  conda run -n analysis python -m ruff check src tests
  ```
- [ ] Run frontend checks from `web/`:
  ```powershell
  Push-Location web
  # User-approved for this N7 implementation plan only: install web dependencies
  # under web/ when node_modules is missing. Do not install outside web/.
  if (-not (Test-Path "node_modules")) {
    npm install
  }
  npm test -- --run --maxWorkers=1
  npm run lint
  npm run build
  Pop-Location
  ```
- [ ] Run whitespace/path sanity:
  ```powershell
  git diff --check
  ```
- [ ] After unit/API/frontend verification passes, run the private 2023H1 dry-run first:
  ```powershell
  $env:PYTHONPATH="src"
  conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation `
    --start-date 2023-01-01 `
    --end-date 2023-06-30 `
    --features mom_20d vol_20d avg_amount_20d `
    --labels forward_return_5d forward_return_20d `
    --kernels tensor dataframe `
    --portfolio-modes long_only long_short `
    --dry-run `
    --output-root reports/private_real/factor_evaluation/n7_eval_2023h1_smoke_v1
  ```
- [ ] Inspect dry-run output for input file, date coverage, required columns, 24 expanded combinations, and output root.
- [ ] After dry-run passes, run the private 2023H1 smoke generation only:
  ```powershell
  $env:PYTHONPATH="src"
  conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation `
    --start-date 2023-01-01 `
    --end-date 2023-06-30 `
    --features mom_20d vol_20d avg_amount_20d `
    --labels forward_return_5d forward_return_20d `
    --kernels tensor dataframe `
    --portfolio-modes long_only long_short `
    --output-root reports/private_real/factor_evaluation/n7_eval_2023h1_smoke_v1
  ```
- [ ] Validate the 2023H1 smoke artifact with schema `0.6` validation and low-memory comparison/report commands.
- [ ] Defer the full 2023-2025 canonical generation until the 2023H1 smoke is inspected and accepted.

### 15. Review and Repair Loop

- [ ] After implementation and verification, run a comprehensive review before claiming the branch is complete.
- [ ] Review the full changed surface systematically:
  - [ ] schema `0.6` artifact contract and validation.
  - [ ] dataframe/tensor kernel parity.
  - [ ] A-share adapter and pipeline integration.
  - [ ] API evidence, preview, download, ZIP, and comparison behavior.
  - [ ] frontend table/chart/selector coverage.
  - [ ] Factor Factory integration.
  - [ ] docs and status updates.
- [ ] Treat review findings as implementation work, not notes for later.
- [ ] Fix every confirmed issue directly on the implementation branch.
- [ ] Re-run the narrowest tests that cover each fix.
- [ ] Repeat the review after fixes.
- [ ] Continue the review -> fix -> verify -> review loop until no confirmed issues remain.
- [ ] Only then prepare final handoff, merge, or PR guidance.

## Acceptance Criteria

- [ ] A single A-share evaluation run can contain all first-stage factor/label/kernel/portfolio-mode combinations.
- [ ] Schema `0.6` artifacts validate from manifest to table columns and paths.
- [ ] `portfolio/weights.parquet` is written, previewed, downloaded, and packaged without CSV-only assumptions.
- [ ] Public QRP-owned contracts use `kernel`, not `engine`.
- [ ] DataFrame and tensor kernels produce structurally identical outputs on deterministic fixtures and numerically equivalent floating values within the test tolerance.
- [ ] The frontend can inspect every required table and switch among labels, kernels, modes, features, or combinations as exposed by artifact metadata.
- [ ] `compare_evaluation_sets` is low-memory and suitable for web/API summary comparison.
- [ ] The detailed Evaluation Comparison Report outputs files only and does not enter the main web UI.
- [ ] Factor Factory adapts to N7 run identity without constraining the N7 design around its old summaries.
- [ ] The 2023H1 private smoke run writes the target 24-combination artifact set after all narrow tests pass.

## Branching Recommendation

- Use one implementation branch for the coordinated N7 contract work:
  ```text
  codex/n7-evaluation-run-implementation
  ```
- Do not split schema, runner, API, and frontend into separate branches unless implementation gets too large to review. These pieces share one contract and will create high merge risk if developed independently.
- Subagents may be opened and closed dynamically at any point during execution when there are independent workstreams or review scopes.
- At most 10 subagents should run at the same time; use fewer when file ownership overlaps or integration review becomes the bottleneck.
- If parallel agents are used, split by file ownership:
  - artifact schema/writer/loader/comparison.
  - evaluation runner/kernels/A-share adapter.
  - API/frontend display integration.
  - Factor Factory/docs/tests review.
- The main implementation thread remains responsible for integrating subagent outputs, resolving conflicts, running verification, and completing the review/fix/re-review loop.
