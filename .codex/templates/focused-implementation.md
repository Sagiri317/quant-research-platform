# Focused Implementation

## Task Mode

`focused-implementation`

## Recommended Reasoning Effort

Medium / Medium-High

Use Medium for a single module or docs/rules batch. Use Medium-High when the change is contract-adjacent, touches linked surfaces, or has meaningful regression risk.

## Context Budget

- Start with `AGENTS.md`, the user request, and the files named by the task.
- Use `rg` to locate current patterns and references before editing.
- Read line ranges first; read full files only when local context is insufficient.
- Report candidate expansion before touching files outside allowed scope.

## Allowed Scope

- Files/directories:
- Behaviors/docs to change:
- Narrow tests/checks:

## Forbidden Scope

- Do not change schemas, `manifest.display`, API/frontend behavior, private artifacts, docs/archive, roadmap priority, dependencies, or directory structure unless explicitly approved.
- Do not delete files or run full tests without approval.

## Stop Conditions

- Stop if the required change crosses the approved task surface, changes a public contract, needs a new dependency, or has unclear blast radius.
- Stop if generated/private data, deployment/trading, external API, secrets, or credential handling becomes relevant.

## Checks

- `git status --short --branch`
- Narrow tests or static checks:
- `git diff --check`
- `git diff --name-only`

## Output Format

- Changed
- Not Changed
- Checks Run
- Scope Check
- Residual Risks
- Recommended commit message
