# Direct Main Docs Only

## Task Mode

`direct-main-docs-only`

Use only when the user explicitly approves low-risk docs-only work directly on `main`.

## Allowed Scope

- User explicitly approved direct-main docs-only execution.
- Typo fixes.
- Formatting-only cleanup.
- Broken link repair.
- Low-risk docs clarification.
- Small docs/status synchronization that records already-merged operational truth is allowed when explicitly approved, as long as it does not change roadmap priority, governance rules, public contracts, schema, phase boundary, private-data boundary, or runtime behavior.

## Forbidden Scope

Use `feature/<task-name>` instead of direct `main` for high-impact docs/rules, including:

- `.codex/templates/*`
- `AGENTS.md`
- `.codex/skills/*/SKILL.md`
- `context-routing.md`
- architecture standards
- roadmap priority
- governance, workflow, or contract wording

Do not change source, tests, schemas, `manifest.display`, API/frontend behavior, dependencies, generated artifacts, private data, or runtime behavior.
Do not amend, rebase, squash, force push, or create merge commits.

## Checks

- `git fetch origin`
- `git status --short --branch`
- `git branch --show-current`
- Verify current branch is `main`.
- Verify `main == origin/main`.
- Verify working tree clean before edits.
- `git diff --check`
- `git diff --name-only`
- Targeted doc/search checks for the changed wording.
- Commit and push `main`.
- Output full commit SHA.

## Output Format

- Direct-main approval
- Changed
- Not Changed
- Checks Run
- Scope Check
- Commit SHA, if committed
- Pushed branch: `main`
