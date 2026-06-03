# Docs Sync

## Task Mode

`docs-sync`

## Recommended Reasoning Effort

Low

## Context Budget

- Read changed docs, directly linked active docs, and the relevant routing note only.
- Use `rg` to find stale names, links, or duplicated claims before opening more files.
- Use `documentation-impact-gate.md` with candidate docs only.
- Do not run an exhaustive Markdown scan unless full docs sync or docs topology cleanup is explicitly requested.
- Do not read archive docs unless the sync explicitly concerns archive references.
- Report candidate expansion before touching code maps, status docs, or architecture docs outside scope.

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

- Docs in scope:
- References to synchronize:
- Wording constraints:

## Forbidden Scope

- Do not change source code, tests, schemas, 0.1 report/API display contract, API/website UI behavior, roadmap priority, private artifacts, or historical docs.
- Do not lower private data, trading, deployment, external API, Conda, Git, or deletion boundaries.

## Stop Conditions

- Stop if the docs disagree with current code/contracts and the correct source of truth is unclear.
- Stop if a sync would change public contract meaning, roadmap priority, current phase, or private-data boundary.

## Verification Honesty

- Do not imply pytest, ruff, or other executable checks passed when they were not run.
- Report pytest/ruff as `not run; docs/rules-only` unless executable code changed.

## Checks

- `rg "<old|new term>" <docs in scope>`
- For touched active docs, report before/after line count. If a touched active doc grows by more than 20 lines or more than 15%, compact an older section in the same commit or justify the growth in the final report.
- `git diff --check`
- `git diff --name-only`
- `git status --short --branch`

## Output Format

- Changed
- Not Changed
- Candidate docs
- Docs checked
- Docs updated
- Docs deferred
- Docs not checked
- Exhaustive docs scan status
- Before/after line counts for touched active docs
- Context Loaded
- Checks Run
- Scope Check
- Recommended commit message
