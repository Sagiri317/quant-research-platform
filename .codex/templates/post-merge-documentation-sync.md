# Post Merge Documentation Sync

## Task Mode

`post-merge-documentation-sync`

Use after a feature branch has been merged and the user asks to synchronize docs or handoff notes.

## Context Budget

- Start from the merge commit, changed-file list, and user-named docs.
- Use `rg` before opening docs.
- Use `documentation-impact-gate.md`; candidate docs only.
- Do not full-scan Markdown unless full docs sync or docs topology cleanup is explicitly requested.

## Allowed Scope

- Active docs that became stale because of the merged change.
- Status or handoff docs only when the merge changed current topic, phase, milestone, blocker, next step, public readiness, or data/API/Git/Conda/privacy boundary.

## Forbidden Scope

- Do not change source, tests, schemas, `manifest.display`, API/frontend, runtime behavior, dependencies, generated artifacts, private data, or archive placement.
- Do not merge, amend, rebase, force push, or delete branches.

## Stop Conditions

- Stop if the required doc update would change roadmap priority, public contract meaning, schema semantics, phase boundary, private-data boundary, or runtime behavior.
- Stop if candidate docs are unclear and resolving them would require a broad docs scan.

## Checks

- `git status --short --branch`
- `rg "<changed term>" <candidate docs>`
- `git diff --check`
- `git diff --name-only`

## Output Format

- Changed
- Candidate docs
- Docs checked
- Docs updated
- Docs deferred
- Docs not checked
- Exhaustive docs scan status
- Checks Run
- Scope Check
