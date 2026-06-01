# Post Merge Documentation Sync

## Task Mode

`post-merge-documentation-sync`

Use after a feature branch has been merged and the user asks to synchronize docs or handoff notes.

## Context Budget

- Start from the merge commit, changed-file list, and user-named docs.
- Use `rg` before opening docs.
- Use `documentation-impact-gate.md`; candidate docs only.
- Do not full-scan Markdown unless full docs sync or docs topology cleanup is explicitly requested.

## Branch Decision

- Use `feature/<task-name>` unless all are true:
  - current branch is clean `main`;
  - user explicitly approved direct-main docs-only execution;
  - the sync is low-risk;
  - it does not touch roadmap priority, governance/workflow rules, contract wording, architecture standards, context routing, skills/templates, or multi-doc current-status surfaces.
- If touching 2+ active docs, `PROJECT_STATUS.md`, roadmap, code-map, current brief, or root handoff signals, treat as a medium+ docs batch and use `feature/<task-name>` unless the user explicitly says direct-main is approved.
- If the user only says "do docs sync", "sync docs", or "post-merge docs checkpoint", that approves the docs task, not direct-main execution.
- If branch choice is ambiguous, stop and ask or create a feature/docs branch; do not commit directly to `main`.

## Allowed Scope

- Active docs that became stale because of the merged change.
- Status or handoff docs only when the merge changed current topic, phase, milestone, blocker, next step, public readiness, or data/API/Git/Conda/privacy boundary.

## Forbidden Scope

- Do not change source, tests, schemas, `manifest.display`, API/frontend, runtime behavior, dependencies, generated artifacts, private data, or archive placement.
- Do not merge, amend, rebase, force push, or delete branches.

## Stop Conditions

- Stop if the required doc update would change roadmap priority, public contract meaning, schema semantics, phase boundary, private-data boundary, or runtime behavior.
- Stop if candidate docs are unclear and resolving them would require a broad docs scan.

## Verification Honesty

- Do not imply pytest, ruff, or other executable checks passed when they were not run.
- Report pytest/ruff as `not run; docs/rules-only` unless executable code changed.

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
