# Task Brief Full

Use this full template for medium+, large, complex, contract-adjacent, or cross-surface work. For small bounded tasks, prefer the fast-path templates in `.codex/templates/`; for `/goal`, overnight, or review-and-fix-loop work, use `goal-overnight.md`.

## Goal

## Task Mode

Examples: Implementation Task, Docs Task, Decision Task, Review Task, Archive/Context-Routing Task.

Recommended template alternatives:

- `discovery-only.md` for inspect-only work.
- `docs-sync.md` for narrow active-doc synchronization.
- `focused-audit.md` or `focused-implementation.md` for standard bounded work.
- `pre-commit-review.md` for branch review before commit.
- `commit-and-push.md`, `branch-push.md`, or `ff-merge-and-push.md` for Git operations.
- `goal-overnight.md` for approved high-token long-run work.

## Context Budget

- Default file budget:
- `rg` searches planned:
- Files to read as snippets:
- Files to read in full:
- Candidate expansion that needs reporting first:

## Context Loaded

List the files actually read before planning.

## Context Routing

List why each context file was loaded. Do not load future architecture docs by default for ordinary coding tasks.

## Task Type

## Current Phase

Local Research Platform

## Execution Settings

- Response language: 默认中文回答
- Recommended reasoning effort:
- Branch plan:
- Worktree needed: yes/no + reason
- Subagents allowed: yes/no
- Planned subagent roles:
- Challenge / Objection allowed: yes/no
- Commit message recommendation required: yes/no

## Recommended Reasoning Effort

- Selected effort:
- Reason:

## Execution Envelope

- Approved autonomous execution: yes/no
- Goal:
- Task mode:
- Files/directories in scope:
- Allowed scope:
- Forbidden scope:
- Contracts that must not change:
- Permitted subagent roles:
- Verification commands:
- Stop conditions:
- Final report format:

## Subagent Plan

- Subagents needed: yes/no
- Subagent role:
- Assigned scope:
- Context packet:
- Write scope disjoint from other subagents: yes/no
- Review-only subagent modifies files: no
- Shared contract owner:
- Expected output:

Codex may use multiple subagents inside an approved envelope. Each subagent must receive a narrow context packet, subagents should have disjoint write scope, review-only subagents must not modify files, and shared contracts must be owned by the main agent or one explicit contract integrator.

## Documentation Sync Plan

- Changed surface:
- Active docs to check:
- Code-map update needed: yes/no
- PROJECT_STATUS update needed: yes/no
- README update needed: yes/no
- Out-of-scope docs follow-up:

## Archive Candidate Check

- New or completed docs created:
- Historical docs touched:
- Archive candidates expected:
- Physical archive allowed by envelope: yes/no

## Allowed Scope

## Forbidden Scope

## Relevant Contracts

- `manifest.display`
- Artifact flow
- Evidence flow

## Files To Inspect

## Expected Changes

## Checks

- Pre-edit checks:
- Narrow verification:
- Final verification:

## Tests Required

## Definition Of Done

## Stop Conditions

Stop for human confirmation if contract meaning, schema semantics, public interfaces, directory moves, phase boundary, or multi-layer scope is unclear.

## Challenge / Objection

If a prompt constraint appears wrong, too narrow, or in conflict with code/docs/tests, output `Proposed Challenge / Objection` before implementing any alternative.

- Challenged constraint:
- Evidence from code/docs/tests:
- Why constraint may be wrong:
- Affected surfaces:
  - schema:
  - `manifest.display`:
  - API/frontend shared contract:
  - data handoff:
  - private-data boundary:
  - memory/performance:
- Recommended alternative:
- Human approval required: yes/no

## Output Format

- Changed
- Not Changed
- Contracts Touched
- Context Loaded
- Subagents Used
- Tests Run
- Documentation Sync
- Archive Candidates
- Recommended commit message
- Commit type: normal commit/amend
- Push target: feature branch only/main
- Residual Risks
- Scope Check

## Risks
