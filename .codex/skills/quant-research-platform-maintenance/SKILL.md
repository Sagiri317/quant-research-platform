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
4. If the change touches `docs/code-map/platform-flow.md` or `docs/code-map/platform-flow-interactive.html`, read [references/code-map-flow.md](references/code-map-flow.md).
5. Keep `PROJECT_STATUS.md` and workspace-root handoff rules as defined by `AGENTS.md`; do not edit root workspace files unless explicitly asked.

Use PowerShell `Select-String` if `rg` is unavailable or denied.

## Contract Map

| Surface | Start here |
| --- | --- |
| Evaluation artifact schema `0.6` | `src/quant_research_platform/artifacts/artifact_schema.py`, `artifact_writer.py`, `artifact_loader.py`, `manifest_validation.py` |
| `manifest.display` | `src/quant_research_platform/artifacts/display_manifest.py`, `manifest_validation.py` |
| Artifact ZIP packaging | `src/quant_research_platform/artifacts/artifact_package.py`, `tests/artifacts/test_artifact_package.py` |
| Evidence API and downloads | `src/quant_research_platform/interfaces/api/routes/`, `interfaces/api/services/`, `interfaces/api/app.py`, `interfaces/api/run_index.py` |
| A-share runtime paths | `src/quant_research_platform/markets/implementations/a_share/runtime/storage.py`, `runtime/check_processed_space.py` |
| A-share private evaluation | `src/quant_research_platform/markets/implementations/a_share/evaluation/run_factor_evaluation.py`, `evaluation/evaluation_adapter.py` |
| A-share Pipeline Builder API | `src/quant_research_platform/interfaces/api/routes/pipeline.py`, `interfaces/api/services/pipeline_service.py`, `src/quant_research_platform/pipelines/a_share/` |
| Factor Factory schema `0.1` | `src/quant_research_platform/research/factor_factory/`, `src/quant_research_platform/interfaces/api/routes/factor_factory.py` |
| N8 contract surfaces | display preview/download helpers, factor-template route/service split, A-share service entry, CSV upload legacy/dev boundary |
| Frontend shell | `web/src/shell/App.tsx`, `web/src/shell/App.test.tsx` |
| Frontend evidence workbench | `web/src/features/evidence/ResultsView.tsx`, `web/src/features/evidence/RunCharts.tsx`, `web/src/features/evidence/ArtifactDownloads.tsx`, `web/src/features/evidence/CsvBrowser.tsx`, `web/src/features/evidence/runEvidenceState.ts`, `web/src/features/evidence/useRunEvidence.ts` |
| Frontend creation surfaces | `web/src/features/uploads/`, `web/src/features/factor-studio/`, `web/src/features/factor-factory/`, `web/src/features/pipeline/`, `web/src/features/runs/` |
| Frontend shared contracts | `web/src/shared/api/api.ts`, `web/src/shared/factorDraft.ts`, `web/src/shared/ui/components.tsx`, `web/src/shared/downloads.ts`, `web/src/shared/format.ts` |
| Local scripts | `scripts/check_all.ps1`, `scripts/start_private_workbench.ps1`, `scripts/start_api_private_runs.ps1`, `scripts/start_web.ps1`, `scripts/stop_workbench.ps1`, `scripts/private_run_roots.ps1`, `scripts/playwright_cli_project.ps1`, `scripts/phase1_acceptance.py`, `scripts/verify_code_map_flow.ps1` |
| Docs and maps | `README.md`, `ARCHITECTURE.md`, `EVALUATION_INTERFACE.md`, `RESEARCH_CARD.md`, `PROJECT_STATUS.md`, `docs/architecture/`, `docs/code-map/` |

## Operating Rules

- Evaluation runs are `artifact_type="evaluation_set"` with `artifact_schema_version="0.6"`.
- Factor Factory runs are `artifact_type="factor_factory_run"` with `artifact_schema_version="0.1"`.
- Do not conflate the two schema versions; tests, API validation, docs, and frontend types must name the artifact type they are checking.
- `manifest.display` remains the only UI/download exposure contract for run artifacts.
- Artifact ZIP assembly belongs in `artifacts/artifact_package.py`; API download services should orchestrate responses instead of owning artifact packaging rules.
- Preserve API-visible evaluation run identity as `(source_alias, run_id)` across run lists, evidence routes, CSV preview, direct downloads, ZIP cache paths, comparison payloads, and frontend route state.
- Pipeline Builder defaults to snapshot creation through `POST /api/pipeline-snapshots`. Executable A-share pipeline requests use `POST /api/pipeline-runs` with `start_from`; every executable run continues through evaluation.
- Factor Studio and Pipeline Builder must keep A-share enabled and crypto/other markets disabled or planned until real modules exist.
- Frontend dependency direction is `shell -> features -> shared`; `web/src/shared/*` must not import from `features/*` or `shell/*`, and cross-feature imports should stay rare.
- For N5/N6 and later research-contract work, reuse the N8 display-helper, factor-template, A-share service-entry, and CSV-upload boundary cleanup surfaces. Do not re-open or replace them inside a research-contract branch unless a written plan states the exact shared files, tests, and rollback path.
- Generated runs under `reports/private_real/`, `reports/runs/`, and `reports/profiles/` are evidence, not authority docs. Regenerate stale runs instead of weakening loaders.

## Contract Work Loop

1. Define invariants first: artifact type, schema version, table names, required columns, `combination_id` semantics, `manifest.display` tables/downloads/presets/charts, API-visible identity, and low-memory behavior.
2. For every artifact table, define the field list, grain, unique key / sort key, status fields, and missing-input semantics together. Do not accept a table contract where one of those pieces is only implied by prose.
3. Treat code schema registries and tests as the source of truth for field counts, required/conditional/optional totals, `required_if` / `nullable_if` rules, and generated file indexes. Markdown may explain the contract, but should not remain the only place where counts or keys are maintained.
4. Map every enforceable invariant to pytest, Vitest, or an existing static check before relying on review text. Add positive and negative coverage when code changes are in scope.
5. Update producer, writer, loader validation, lightweight run index, API payload/download behavior, frontend consumers, tests, docs, and code maps in the same round when the contract changes.
6. Remove retired compatibility in one pass when old generated runs are no longer supported. Search for stale schema versions, retired summary fields, old table names, and frontend-only shortcuts.
7. Review the diff by surface: artifact contract, API/downloads, frontend UI, A-share pipeline/runtime, Factor Factory, tests/fixtures, docs/skills, generated outputs, and gitignore.
8. For docs-only drift, verify against current code paths before preserving command examples or path names.

## Rule Placement

- Put hard project boundaries in `AGENTS.md`: private data, Conda/Git/deletion safety, `(source_alias, run_id)`, `manifest.display`, frontend dependency direction, and schema-change synchronization.
- Put executable contract checks in pytest or Vitest whenever possible. This includes required artifact files/columns, display coverage, lightweight-index rejection, preview limits, download gating, retired public fields, and frontend display parity.
- Put workflow reminders in this skill: what surfaces to review, which checks to run, and when browser/private acceptance is appropriate.
- Put human-readable contract explanations in root docs, architecture docs, and code maps. Do not rely on docs or skills as the only enforcement for a behavior that tests can verify.
- Add a new script only when repeated static checks are too easy to miss manually; prefer tests first and keep scripts as orchestration or hygiene helpers.

## Review Checklist

For artifact/API/frontend contract work, explicitly review these invariants before calling the branch ready:

- Schema required files and columns are enforced by both the full loader and any lightweight index/list path that makes runs API-visible.
- Schema field counts, directory totals, required/conditional/optional status, table grain, unique keys, sort keys, and registry appendix entries agree with each other.
- Required-if, nullable-if, sentinel, and family-specific validation rules are represented in schema registry/tests rather than only in prose.
- `manifest.display` remains the only exposure source for evidence payloads, table previews, direct downloads, ZIP downloads, frontend tables, charts, and download presets.
- `manifest.display.tables[].path` and download item paths resolve only to run-relative paths declared in `manifest.artifacts`.
- Preview endpoints have bounded defaults, hard maximum limits, and clear low-memory metadata such as `has_more` and whether row counts are exact.
- `manifest.display.tables[].key` values are unique. Duplicate keys must fail artifact validation because table lookup, API preview, frontend state, and downloads all treat the key as a single contract identity.
- Download services package files from manifest-resolved artifact paths and do not full-load Parquet tables for metadata-only or selected-file downloads.
- Public-downloadable metadata such as `config.json` and `lineage.json` is sanitized consistently at writer, direct-download, and ZIP-packaging paths, including selected ZIP cache fingerprints.
- Frontend selected-run and selected-combination state drives fresh table/chart previews when the initial evidence payload is intentionally truncated.
- Frontend summary cards must not infer "latest" rows from a truncated preview page or from non-exact row counts. When a card needs the last row for a filtered `combination_id`, use bounded preview pagination such as `limit=1&offset=total_rows-1` only when the preview payload marks `total_rows_exact=true`; otherwise make the UI copy explicitly say it is showing a preview slice or do not render a latest-row claim.
- Frontend table and chart preview requests must respect API-provided `preview.max_limit` / `previewMaxRows`. Do not hardcode a larger limit, including for chart tables.
- Chart surfaces that need a complete selected-combination time axis should page through the preview API with bounded `limit` plus `offset` / `has_more`; CSV/detail table previews should keep the display default row limit and point users to downloads for full rows.
- Generated schema-`0.6` runs that lack current `manifest.display.tables[].preview` metadata are stale. API list/evidence/preview entry points must skip or reject them through tests instead of weakening validation.
- Factor Factory consumes the current artifact/API contract without forcing N7 evaluation artifacts to preserve retired fields.
- Retired public fields and table names are absent from current `src/` and `web/` contracts; allowed historical references and negative tests must be intentional.
- Private/generated outputs remain ignored and untracked.

## Review Stop Rules

- Fix confirmed P0/P1/P2 issues in the same round when they affect the requested contract, correctness, memory safety, private-data safety, or user-visible behavior.
- Do not expand the release-blocking scope for P3 cleanup, future hardening, or historical documentation that is intentionally marked as old context. Record it as residual risk when it is not part of the current change.
- For planning specs, run at most one broad multi-perspective review and one cold-start review before moving to implementation planning. If additional issues keep appearing after that, convert the contract into schema registry/tests instead of expanding Markdown prose.
- Use three independent review lenses for broad schema work: schema/field consistency, engineering/API/display contract, and quant research semantics. Avoid repeated local-only review loops that keep finding the same synchronization class.
- After each fix, run the narrowest relevant test or static check and do a focused re-review of the touched surface.
- After route/service entry splits, update wrapper tests to monkeypatch the symbol that the tested module actually resolves, usually the imported service entry in the wrapper module. Do not keep tests patched to retired lower-level symbols just because they existed before the split.
- Do one final cold-start review after the relevant broad verification passes, or earlier if broad verification exposes a new failure domain.
- Stop the review loop only when enforceable invariants have tests or static checks, relevant verification passes, one final review finds no confirmed blocker, and private/generated outputs remain untracked.
- If new issues keep appearing in the same surface, convert the repeated reviewer question into a test or static check before doing another manual pass.

Use [references/contracts-and-verification.md](references/contracts-and-verification.md) for severity definitions, examples, and when focused review is enough versus a broader re-review.

## Verification Profiles

Run the narrowest relevant checks first. For broad contract work, prefer the relevant subset rather than a single unconditional full-suite command:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts tests/interfaces/api
conda run -n analysis python -m pytest -q tests/evaluations tests/pipelines/a_share tests/markets/implementations/a_share/evaluation tests/research/factor_factory
conda run -n analysis python -m ruff check src tests
Push-Location web; try { npm test -- --run --maxWorkers=1; npm run lint; npm run build } finally { Pop-Location }
git diff --check
```

For source/static hygiene, use targeted searches when relevant instead of creating a separate gate script by default. Search for retired fields, private/local-only inputs leaking into API/frontend/pipeline surfaces, API full-load hot paths, and untracked generated outputs. Promote any recurring static check into tests or a lightweight script only when manual repetition becomes error-prone.

Use `scripts/phase1_acceptance.py` only for Phase 1 private acceptance tasks that intentionally touch `reports/private_real/`: `rename-baseline`, `run`, `compare`, or `factory-smoke`. It is not a routine unit-test substitute, and it should not be used for public-safe or source-only documentation checks.

For frontend evidence-workbench changes, run browser acceptance only when the UI behavior changed. For API-backed workbench flows, start both backend and frontend services with `scripts/start_api_private_runs.ps1` and `scripts/start_web.ps1`, then stop them with `scripts/stop_workbench.ps1`. Use `scripts/playwright_cli_project.ps1` for Playwright CLI work in this repository so screenshots, traces, PDFs, videos, and binary response bodies land under ignored `.playwright-cli/` instead of a top-level `output/` directory. List any logs, downloads, screenshots, or cache outputs before cleanup. Follow project deletion rules exactly.

## Reference Loading

Read [references/contracts-and-verification.md](references/contracts-and-verification.md) when you need detailed exit criteria for:

- evaluation-set schema `0.6`;
- Factor Factory schema `0.1`;
- API/download validation;
- frontend state and browser acceptance;
- A-share pipeline/runtime boundaries;
- Phase 1 acceptance script usage;
- skill-governance and global-skill sedimentation recommendations.

Read [references/code-map-flow.md](references/code-map-flow.md) when changing or reviewing `docs/code-map/platform-flow.md` or `docs/code-map/platform-flow-interactive.html`, especially node/edge coverage, API/frontend relationship direction, interactive edge editing, or MD/HTML synchronization.
