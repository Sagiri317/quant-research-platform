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
4. If the change touches schema/artifact contracts, `manifest.display`, API/download/frontend exposure, module boundaries, dependency direction, testing layers, generated outputs, scale, or security boundaries, read `docs/architecture/engineering-standards.md`.
5. If the change touches schema `0.7` diagnostics, diagnostic tables, diagnostic UI/report wording, factor evaluation semantics, diagnostics interpretation, tradability, costs, liquidity, capacity, benchmarks, risk, lineage, or paper/live trading wording, read `docs/architecture/quant-research-standards.md`.
6. If the change touches `docs/code-map/platform-flow.md` or `docs/code-map/platform-flow-interactive.html`, read [references/code-map-flow.md](references/code-map-flow.md).
7. Keep `PROJECT_STATUS.md` and workspace-root handoff rules as defined by `AGENTS.md`; do not edit root workspace files unless explicitly asked.

Use PowerShell `Select-String` if `rg` is unavailable or denied.

For medium+ tasks, stop before implementation with a task brief, implementation plan, scope declaration, and explicit human approval. Keep the summary format concrete: Changed, Not changed, Risks, Scope check.

## Stability Model

Keep this `SKILL.md` as the stable workflow layer. Put change-prone project facts in [references/contracts-and-verification.md](references/contracts-and-verification.md), including schema versions, source-file maps, script names, API route names, browser acceptance inputs, and current verification commands.

Stable responsibilities:

- Preserve project safety boundaries from `AGENTS.md`; do not move task-specific approvals into permanent rules.
- Current phase is Local Research Platform, with Stable Core + High-Velocity Research Edge. Do not add speculative infrastructure or unimplemented topology during maintenance work.
- Keep generated artifacts as the contract source for evidence flows, with `manifest.display` as the UI/API/download exposure boundary.
- Keep artifact types explicit. Do not conflate evaluation-run artifacts, Factor Factory artifacts, generated evidence, and historical/rejection fixtures.
- Keep schema registries and tests as the source of truth for field lists, grain, unique keys, conditional rules, and required generated files.
- Prefer linked updates across writer, loader, `manifest.display`, API payloads/routes, downloads, frontend consumers, tests, docs, and code maps when a public contract changes.
- Cross-check architecture, engineering, and quant research standards before changing long-term module boundaries, artifact/API/frontend exposure, diagnostics interpretation, or trading-adjacent wording.
- Treat generated runs as evidence, not authority docs. Regenerate stale runs instead of weakening validation to keep them visible.
- Active docs should contain current operational truth only: current status, blockers, next steps, handoff, and valid samples.
- `docs/code-map/` is the Current Implementation Learning Layer: current implementation reality, runtime flow, entrypoints, module relations, artifact/API/frontend tracing, and onboarding. Do not put roadmap, governance, unimplemented architecture planning, project status, historical closeout, or research narrative there.

Before using concrete paths, scripts, schema numbers, or route names, load the current contract reference.

## Contract Work Loop

1. Define invariants first: artifact type, schema version, table names, required columns, `combination_id` semantics, `manifest.display` tables/downloads/presets/charts, API-visible identity, and low-memory behavior.
2. For every artifact table, define the field list, grain, unique key / sort key, status fields, and missing-input semantics together. Do not accept a table contract where one of those pieces is only implied by prose.
3. Treat code schema registries and tests as the source of truth for field counts, required/conditional/optional totals, `required_if` / `nullable_if` rules, and generated file indexes. Markdown may explain the contract, but should not remain the only place where counts or keys are maintained.
4. Map every enforceable invariant to pytest, Vitest, or an existing static check before relying on review text. Add positive and negative coverage when code changes are in scope.
5. Update writer, loader validation, `manifest.display`, lightweight run index, API payload/download behavior, frontend consumers, tests, docs, and code maps in the same round when the contract changes.
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

For artifact/API/frontend contract work, review the stable invariant classes here, then load the current reference for exact schema names, files, and commands:

- Required files, required columns, field counts, grain, unique keys, sort keys, conditional rules, and generated-file indexes are enforced by registry-backed tests.
- `manifest.display` is the only exposure source for evidence payloads, previews, downloads, frontend tables, charts, and presets.
- Lightweight list/index paths reject stale or invalid artifacts before making them API-visible.
- Preview/download paths are bounded, display-approved, path-safe, and avoid full-loading large tables for metadata-only operations.
- Frontend state follows API/display contracts for selected run, selected combination, table previews, charts, reports, and downloads.
- Retired public fields, routes, table names, and compatibility wrappers are either absent from current contracts or explicitly covered as historical/rejection tests.
- Private paths, secrets, task-local aliases, generated outputs, and row-level private data do not leak into portable artifacts, API metadata, logs, downloads, or commits.

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

Run the narrowest relevant checks first. Load [references/contracts-and-verification.md](references/contracts-and-verification.md) for the current command set and acceptance wrappers.

Stable verification rules:

- Prefer focused pytest/Vitest/static checks for the touched contract before broad verification.
- Use targeted searches for retired fields, private/local-only inputs leaking into API/frontend/pipeline surfaces, API full-load hot paths, and untracked generated outputs.
- Promote recurring static checks into tests or lightweight scripts only when manual repetition becomes error-prone.
- For browser acceptance, start both API and frontend services, record service evidence, stop only recorded services, and list generated browser artifacts before cleanup.
- Always include `git diff --check`; include `git status --short` when generated, private, browser, or acceptance outputs may have changed.

## Reference Loading

Read [references/contracts-and-verification.md](references/contracts-and-verification.md) when you need detailed exit criteria for:

- current evaluation-set artifact schema;
- current Factor Factory artifact contract;
- API/download validation;
- frontend state and browser acceptance;
- A-share pipeline/runtime boundaries;
- generated-run acceptance and cleanup rules;
- skill-governance and global-skill sedimentation recommendations.

Read [references/code-map-flow.md](references/code-map-flow.md) when changing or reviewing `docs/code-map/platform-flow.md` or `docs/code-map/platform-flow-interactive.html`, especially node/edge coverage, API/frontend relationship direction, interactive edge editing, or MD/HTML synchronization.
