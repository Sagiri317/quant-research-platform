# Goal Overnight

## Task Mode

`goal-overnight`

Covers `/goal`, overnight runs, and review-and-fix-loop work.

## Recommended Reasoning Effort

High / Highest available

Goal mode can use high token budgets when the approved envelope justifies them. High-token mode is for maximizing closed engineering loops under Pro limits, not for unconstrained context loading.

## Approved Execution Envelope

- Approved autonomous execution: yes/no
- Goal and why it matters:
- Task mode: `/goal` / overnight / review-and-fix-loop
- Files/directories in scope:
- Allowed scope:
- Forbidden scope:
- Contracts that must not change:
- Permitted subagent roles:
- Verification commands:
- Stop conditions:
- Final report format:

## Context Budget

- Context loading plan:
- Start with `AGENTS.md`, this template, the maintenance skill, context routing, and task-specific current docs.
- Use `rg` before opening additional files.
- Read assigned files, relevant diffs, and task-specific contracts before broader docs.
- Broader context is allowed only inside this approved envelope.
- Report candidate expansion before loading architecture, archive, generated evidence, private data, or unrelated modules not named in the envelope.

## Goal And Why It Matters

- Goal:
- User impact:
- Engineering closure target:
- Non-goals:

## Allowed Scope

- Files/directories:
- Fix classes allowed:
- P0/P1/P2 handling: confirmed P0/P1/P2 issues inside the approved envelope may be fixed without per-fix approval.
- P3 handling: report P3 cleanup, or fix only when small, clearly in scope, and low risk.

## Forbidden Scope

- Do not remove private data boundaries.
- Do not change schemas, `manifest.display`, API/frontend shared contracts, deployment/trading scope, market/frequency/simulation scope, dependencies, directory structure, archive placement, or unclear blast-radius surfaces unless the envelope explicitly approves that exact change.
- Do not delete files, generate private artifacts, expose private data, install dependencies, or run full tests unless approved.
- For any approved CLI, artifact writer, exporter, download, ZIP/package,
  metadata writer, or persisted row-level output slice, apply the maintenance
  skill `Private Artifact Writer Checklist`.

## Phase Plan

- Phase 0: Confirm envelope, branch/worktree state, and stop conditions.
- Phase 1: Load narrow context through context routing.
- Phase 2: Run `pre-goal-blocker-discovery.md`.
- Phase 3: If `Can proceed` is yes, audit or implement the first bounded slice.
- Phase 4: Run narrow checks and fix confirmed in-scope P0/P1/P2.
- Phase 5: Repeat review-and-fix loop until no confirmed blocker remains or a stop condition is hit.
- Phase 6: Run final approved verification and produce final report.

## Pre-Goal Blocker Discovery

- Can proceed: yes/no
- Blockers:
- Escalations:
- Safe implementation slice:
- Files likely to change:
- Tests likely to run:

If `Can proceed` is no, do not edit files.

## Subagent Plan

- Subagents allowed: yes/no
- Roles:
- Assigned files/directories or diff slices:
- Context packet per subagent:
- Write scopes are disjoint: yes/no
- Shared contract owner: main agent / named integrator
- Review-only subagents modify files: no
- For contract/safety/artifact-adjacent work, assign subagents by issue family
  instead of generic review: ordering/preflight/duplicate-key,
  identifier/path/private-marker sanitizer, artifact freshness / generated
  artifact cleanup, API/CLI/service contract, and merge-readiness / test
  coverage.

## Issue-Family Checklist

For contract-adjacent, safety-adjacent, artifact-adjacent, metadata writer,
CLI/API/service, schema inventory, EDA refresh, run discovery, or private-output
boundary goals, list issue families before implementation or review. Do not
only fix the file where the bug was observed; scan sibling entrypoints for the
same issue family.

Issue families to check:

- duplicate-before-input-read / duplicate-before-row-level-write
- preflight-before-row-read
- unsafe identifier before metadata/API/CLI echo
- sensitive text and private-marker handling in metadata keys and values
- schema inventory redaction for private/vendor/source/local markers
- stale generated artifact refresh / cleanup
- disk-loaded metadata numeric/string sanitization
- CLI/API/service failure-code consistency
- configured root / run discovery consistency
- long/wide artifact consistency
- deterministic seed / optional dependency behavior
- private input + public output boundary
- row-level private-derived artifact tracking

Include a blast-radius declaration across service, CLI, API, schema/metadata
writer, artifact writer, EDA refresh, run discovery, docs/status, tests, and
private-output boundary before changing code.

Require parameter-family RED tests where implementation changes are allowed:
unsafe identifier variants (`vendor_factor`, `vendorFactor`, `factorVendor`,
`sourceAlpha`, `localTimestamp`, `../alpha`, `C:/tmp/alpha`, `..\alpha`),
allowlist regressions (`forward_return_1d`, `walk_forward`,
`random_forest_lite`), duplicate keys across service/API/CLI when applicable,
malformed metadata across status/schema inventory/EDA/comparison, and optional
dependencies in installed and missing paths.

Wording constraints: use generic reusable wording. Do not include real commit
SHAs, private paths, row samples, stock-code samples, alpha/performance/
investment claims, or a long historical narrative.

## Review-And-Fix Loop

- Review lens:
- Severity standard:
- Fix threshold: P0/P1/P2 inside approved scope.
- P3 policy: report or fix only if small and in scope.
- Re-check command after each slice:
- Stop when repeated findings require a new test/static check or exceed approved scope.
- Final review gate: confirm every issue family was checked across relevant
  entrypoints; tests are parameterized by variants, not only smoke cases;
  API/CLI/service error codes are consistent; disk-loaded metadata paths are
  sanitized; generated artifacts are ignored/untracked; and docs are precise
  without overclaiming.

## Verification Cadence

- After each slice:
- Before commit:
- Before push:
- Before merge:
- Full-suite approval required: yes/no

## Checkpoint Report Cadence

- Report after each phase or every:
- Include: completed slice, files changed, checks run, next slice, stop-condition status.

## Commit / Push / Merge Policy

- Branch tasks use `feature/<task-name>`.
- Commit and merge are not bound together.
- After a commit, push the committed branch unless the envelope says not to.
- After a merge, push the merged target branch.
- No amend, rebase, squash, merge commit, branch deletion, or force push unless explicitly approved.
- For fast-forward to `main`, use `ff-merge-and-push.md`.
- Do not merge `main` automatically unless the envelope explicitly allows it.

## Stop Conditions

- Stop for schema change, `manifest.display` change, public API/frontend shared contract change, directory restructuring, new dependency, new abstraction/framework, market/frequency/simulation/trading scope expansion, private/generated data edits, file deletion, archive or historical-doc physical moves, and unclear blast radius.
- Stop for secrets, credentials, external API/vendor/broker/exchange connections, deployment exposure, or private data leakage risk.
- Stop if verification failures point outside the approved envelope.

## Checks

- Envelope-approved checks:
- Narrow checks after each slice:
- `git status --short --branch`
- `documentation-impact-gate.md` result before final report
- `git diff --check`
- `git diff --name-only`

## Output Format

- Changed
- Not Changed
- Contracts Touched
- Context Loaded
- Subagents Used
- Checks Run
- Review-And-Fix Loop Summary
- Documentation Sync
- Documentation Impact
- Archive Candidates
- Stop Conditions
- Residual Risks
- Scope Check
- Recommended commit message
