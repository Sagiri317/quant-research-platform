---
name: quant-research-platform-maintenance
description: Use when changing quant-research-platform artifact schemas, manifest.display, evidence API, download packaging, React evidence UI, A-share pipeline runs, Factor Factory runs, generated runs, or docs that describe those contracts.
---

# Quant Research Platform Maintenance

## Purpose

Keep generated artifacts as the source of truth, `manifest.display` as the API/UI/download contract, and the React workbench as a consumer of those contracts. Do not restore hidden frontend business logic, schema fallbacks, broad filesystem cleanup, or private-data exposure.

Project `AGENTS.md` owns hard safety boundaries: local-only scope, approved Conda environment, data/API/secrets limits, forbidden operations, private CSMAR boundaries, and editable areas. This skill owns the operational workflow for artifact/API/frontend/A-share/Factor Factory contract changes and the docs that describe them.

The principles behind this workflow are described in `docs/architecture/ai-native-engineering-principles.md`, but actual execution remains governed by this skill and project `AGENTS.md`.

## Start Here

1. Read project `AGENTS.md`.
2. Ordinary QRP tasks may read `.codex/briefs/current-qrp-status.md` as a short current-context primer before broad-loading `PROJECT_STATUS.md`, `README.md`, or the roadmap.
3. Inspect only the requested surface and its linked contract files.
4. If the change touches schema, API payloads, downloads, frontend evidence state, A-share runs, Factor Factory runs, or generated evidence, read [references/contracts-and-verification.md](references/contracts-and-verification.md).
5. If the change touches schema/artifact contracts, `manifest.display`, API/download/frontend exposure, module boundaries, dependency direction, testing layers, generated outputs, scale, or security boundaries, read `docs/architecture/engineering-standards.md`.
6. If the change touches schema `0.7` diagnostics, diagnostic tables, diagnostic UI/report wording, factor evaluation semantics, diagnostics interpretation, tradability, costs, liquidity, capacity, benchmarks, risk, lineage, or paper/live trading wording, read `docs/architecture/quant-research-standards.md`.
7. If the change touches `docs/code-map/platform-flow.md` or `docs/code-map/platform-flow-interactive.html`, read [references/code-map-flow.md](references/code-map-flow.md).
8. For task-specific context selection, use [references/context-routing.md](references/context-routing.md). Keep it as routing guidance, not an additional approval layer.
9. Keep `PROJECT_STATUS.md` and workspace-root handoff rules as defined by `AGENTS.md`; the current brief is not a replacement for `PROJECT_STATUS.md`, and changes to current topic, blocker, milestone, next step, public readiness, data/API/Git/Conda/privacy boundary, or handoff still require checking `PROJECT_STATUS.md`.

Use PowerShell `Select-String` if `rg` is unavailable or denied.

For medium+ tasks, stop before implementation with a task brief, implementation plan, scope declaration, and explicit human approval. Use the current final report format: Changed, Not Changed, Contracts Touched, Context Loaded, Subagents Used, Tests Run, Residual Risks, Scope Check.

## Execution Settings

Default response language:

- Default: 默认中文回答.
- Answer in Chinese by default.
- Code, file names, field names, commands, test names, error messages, schema/table/metric names, and section labels such as Changed, Not Changed, Tests Run, and Scope Check may remain in English.
- This language rule applies to Codex conversation output only. It does not force code comments, user-facing docs, or project documents to use Chinese.

Recommended reasoning effort:

- Low: typo fixes, single-file docs cleanup, and formatting.
- Medium: small docs/rules batches, local bugfixes, and single-module implementation.
- High: cross-layer code, diagnostics, artifact/API/frontend work, pipeline work, memory/performance work, and contract-adjacent work.
- Highest / max available: architecture decisions, schema migrations, major refactors, and future capability boundaries.
- Use the closest available Codex reasoning-effort setting.

Task modes and template selection:

| Class | Task mode | Template | Recommended reasoning effort | Default context posture |
| --- | --- | --- | --- | --- |
| Router | `auto-task-router` | `.codex/templates/auto-task-router.md` | Low | Use only when the user gives a short task intent instead of an expanded prompt. |
| Fast Path | `discovery-only` | `.codex/templates/discovery-only.md` | Low | Inspect only; no edits. |
| Fast Path | `docs-sync` | `.codex/templates/docs-sync.md` | Low | Changed docs and direct active links only. |
| Fast Path | `direct-main-docs-only` | `.codex/templates/direct-main-docs-only.md` | Low | Only explicitly approved low-risk docs-only work on `main`. |
| Fast Path | `commit-and-push` | `.codex/templates/commit-and-push.md` | Low | Current branch status and staged/unstaged diff only. |
| Fast Path | `branch-push` | `.codex/templates/branch-push.md` | Low | Current branch and local commits only. |
| Fast Path | `ff-merge-and-push` | `.codex/templates/ff-merge-and-push.md` | Low | Approved feature branch, `main`, and merge status only. |
| Fast Path | `branch-cleanup` | `.codex/templates/branch-cleanup.md` | Low | Explicitly approved single branch only. |
| Standard | `focused-audit` | `.codex/templates/focused-audit.md` | Medium / High | Target surface, relevant diff, and linked contracts only. |
| Standard | `focused-implementation` | `.codex/templates/focused-implementation.md` | Medium / Medium-High | Target files and narrow tests first. |
| Standard | `pre-commit-review` | `.codex/templates/pre-commit-review.md` | Medium | Diff-only by default. |
| Standard | `documentation-impact-gate` | `.codex/templates/documentation-impact-gate.md` | Medium | Candidate docs only; no exhaustive docs scan by default. |
| Goal / Overnight | `pre-goal-blocker-discovery` | `.codex/templates/pre-goal-blocker-discovery.md` | High | Decide whether long-run implementation can proceed before editing. |
| Goal / Overnight | `goal-overnight` | `.codex/templates/goal-overnight.md` | High / Highest available | Broader context only inside an approved envelope. |

Use Fast Path modes for bounded routine work so ordinary tasks do not accidentally become broad audits. Use Standard modes when edits or review need focused reasoning over a task surface. Use Goal / Overnight only when the user has approved the full execution envelope required by `AGENTS.md`.

Template loading modes:

- Expanded Prompt Mode: if the user provides a complete task brief or approved envelope, follow it and load only named or directly relevant files.
- Repo Router Mode: if the user gives only a short task intent, start from `.codex/templates/auto-task-router.md`, then load the smallest selected template set.
- Do not full-load all templates, all docs, generated outputs, private data, or archives by default.

Preserved workflow controls:

- Default conversation language remains Chinese unless the user asks otherwise.
- Every task template must recommend a reasoning effort instead of asking the user to switch models frequently.
- Challenge / Objection, commit message recommendation, subagent policy, Scope Check, Stop Conditions, Documentation Sync Gate, and Archive Candidate Gate remain active.
- Goal / Overnight mode may use high token budgets when approved, but it does not weaken private data, schema, `manifest.display`, API/frontend, deployment/trading, file deletion, archive movement, or unclear blast-radius stop conditions.

Branch / Worktree policy:

- Small docs-only, typo, and single-file rules cleanup may run on a clean `main`.
- Branch tasks use `feature/<task-name>`.
- Do not default to `codex/`, `docs/`, or `chore/` branch prefixes.
- Medium+ docs/rules batches should use `feature/<task-name>`.
- Code implementation, tests, diagnostics, pipeline, Factor Factory, API, and artifact work must use `feature/<task-name>` instead of implementing directly on `main`.
- Contract-adjacent, schema-adjacent, and cross-layer work must use `feature/<task-name>`.
- Direct `main` docs-only execution is allowed only when explicitly approved and low risk; high-impact workflow, governance, template, skill, context-routing, architecture standard, roadmap priority, or contract wording changes still require a branch.
- Use a worktree only when multiple tasks run in parallel, the current workspace has uncommitted changes that must not be interrupted, a long-running goal needs isolation, or the current `main` / workspace state must be preserved.
- Do not create a worktree by default.
- Every task brief should state the branch plan and `Worktree needed: yes/no + reason`.

Closed-loop execution:

- When the user approves a bounded envelope, Codex may run plan, implementation, review/fix, verification, commit, and feature-branch push without repeated confirmation.
- Stop only for blocker or escalation items: new dependency/framework, schema or `manifest.display`, public API/frontend shared contract, new artifact type or required artifact file, new top-level package, A-share adapter capability expansion, model registry, feature store, scheduler, worker, deployment, external API/vendor/broker/exchange connector, paper/live trading, private path or row-level data risk, roadmap priority change, dirty working tree or unclear base branch, repeated unclear test failure, or unclear blast radius.
- Do not automatically merge `main` unless the approved envelope explicitly allows it.

Challenge / Objection mechanism:

If Codex believes a prompt constraint is wrong, too narrow, or conflicts with code/docs/tests, it must not silently obey and must not silently violate it. Before implementing the alternative, output:

```text
Proposed Challenge / Objection
```

Each objection must include:

- challenged constraint
- evidence from code/docs/tests
- why constraint may be wrong
- affected surfaces: schema, `manifest.display`, API/frontend shared contract, data handoff, private-data boundary, memory/performance
- recommended alternative
- whether human approval is required

Do not implement a challenged alternative until the user approves it.

## Stability Model

Keep this `SKILL.md` as the stable workflow layer. Put change-prone project facts in [references/contracts-and-verification.md](references/contracts-and-verification.md), including schema versions, source-file maps, script names, API route names, browser acceptance inputs, and current verification commands.

Stable responsibilities:

- Preserve project safety boundaries from `AGENTS.md`; do not move task-specific approvals into permanent rules.
- Current phase, architecture style, and hard stop boundaries are owned by `AGENTS.md`; load it before applying this workflow.
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

## Goal / Long-Run Workflow

Use this workflow only after the user has approved the bounded execution envelope required by `AGENTS.md`.

1. Confirm the approved envelope and final report format.
2. Load context through [references/context-routing.md](references/context-routing.md).
3. Split work only by disjoint file ownership.
4. Keep shared contracts under the main agent or one explicit contract integrator.
5. Run narrow checks after each slice.
6. Fix confirmed P0/P1/P2 issues that are inside the approved scope without asking again.
7. Stop on `AGENTS.md` stop conditions or when blast radius becomes unclear.
8. End with the approved final report.

## Subagent Coordination Workflow

- Codex may use multiple subagents inside an approved envelope.
- Give each subagent a narrow context packet and disjoint write scope.
- Review-only subagents must not modify files.
- Reviewer subagents should not edit implementation unless explicitly assigned that role.
- Do not let multiple subagents modify schema, API payloads, frontend shared types, `manifest.display`, generated-file indexes, or other shared contracts in parallel.
- The main agent integrates subagent output, checks scope, and verifies that assigned files stayed within the approved envelope.
- Final reports must include `Subagents Used`.

## Documentation Sync Gate

Run this end-of-task checklist before final reporting for medium+ tasks, goals, and overnight runs:

1. Review `git diff --name-only`.
2. Map changed files to the task surface.
3. Check whether linked active docs need sync.
4. Update linked active docs only when they are inside the approved scope.
5. If needed doc sync is outside scope, report it as follow-up / residual risk.
6. Do not update archived docs to current truth unless explicitly requested.

Surface-to-doc map:

- Artifact schema / `manifest.display`: `EVALUATION_INTERFACE.md`, `docs/code-map/evaluation-artifacts-api.md`, contracts reference.
- API/download/evidence: `docs/code-map/evaluation-artifacts-api.md`.
- Frontend evidence: `docs/code-map/evaluation-artifacts-api.md`.
- A-share pipeline: `docs/code-map/a-share-data-and-pipeline.md`.
- Factor Factory: `docs/code-map/a-share-data-and-pipeline.md`; contracts reference if artifact/API contract changed.
- Current status / milestone / blocker / next step: `PROJECT_STATUS.md`.
- Project entry workflow: `README.md`.
- Architecture/future boundary: `docs/architecture/*`.

## Archive Candidate Gate

Run this end-of-task checklist before final reporting. It reports candidates only; it does not authorize moving, deleting, or rewriting archive content.

Archive candidate signals:

- Dated implementation plan / spec / closeout.
- Completed task execution record.
- Old schema or old milestone record.
- Current summary already exists elsewhere.
- Active placement risks misleading future agents.

Physical archive movement is allowed only when the approved envelope explicitly allows archive movement.

Final reports must include:

```text
Archive Candidates:
- path:
- reason:
- active references:
- recommended action:
```

If none, write `Archive Candidates: none`.

## Commit Message Recommendation

For medium+ tasks, the final report should recommend a commit message.

- Match project style, usually Conventional Commit where appropriate: `feat(scope): ...`, `fix(scope): ...`, `docs(scope): ...`, `refactor(scope): ...`, or `test(scope): ...`.
- Do not imply unimplemented capability.
- For proxy or descriptive diagnostics, do not imply true risk, capacity, simulation, or validated alpha.
- Include whether the commit should be a normal commit or amend.
- Include whether push should be feature branch only or `main`.

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

Read [references/context-routing.md](references/context-routing.md) when deciding which project docs to load for a task. It is a short routing matrix only; it does not replace `AGENTS.md`, this skill, or current code/tests.
