# AI Task Brief Template

## Goal

## Task Mode

Examples: Implementation Task, Docs Task, Decision Task, Review Task, Archive/Context-Routing Task.

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

## Final Report Format

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
