# Docs Sync

## Task Mode

`docs-sync`

## Recommended Reasoning Effort

Low

## Context Budget

- Read changed docs, directly linked active docs, and the relevant routing note only.
- Use `rg` to find stale names, links, or duplicated claims before opening more files.
- Do not read archive docs unless the sync explicitly concerns archive references.
- Report candidate expansion before touching code maps, status docs, or architecture docs outside scope.

## Allowed Scope

- Docs in scope:
- References to synchronize:
- Wording constraints:

## Forbidden Scope

- Do not change source code, tests, schemas, `manifest.display`, API/frontend behavior, roadmap priority, private artifacts, or docs/archive.
- Do not lower private data, trading, deployment, external API, Conda, Git, or deletion boundaries.

## Stop Conditions

- Stop if the docs disagree with current code/contracts and the correct source of truth is unclear.
- Stop if a sync would change public contract meaning, roadmap priority, current phase, or private-data boundary.

## Checks

- `rg "<old|new term>" <docs in scope>`
- `git diff --check`
- `git diff --name-only`

## Output Format

- Changed
- Not Changed
- Context Loaded
- Checks Run
- Scope Check
- Recommended commit message
