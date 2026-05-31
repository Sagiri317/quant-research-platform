# Direct Main Docs Only

## Task Mode

`direct-main-docs-only`

Use only when the user explicitly approves low-risk docs-only work directly on `main`.

## Allowed Scope

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

## Checks

- `git status --short --branch`
- `git branch --show-current`
- `git diff --check`
- `git diff --name-only`
- Targeted doc/search checks for the changed wording.

## Output Format

- Direct-main approval
- Changed
- Not Changed
- Checks Run
- Scope Check
- Commit SHA, if committed
