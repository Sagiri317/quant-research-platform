# Pre Commit Review

## Task Mode

`pre-commit-review`

## Recommended Reasoning Effort

Medium

## Context Budget

- Diff-only by default: `git status --short --branch`, `git diff --name-only`, and `git diff`.
- Read changed files only around changed line ranges when needed.
- Use `rg` only for changed symbols or references that affect the diff.
- Report candidate expansion before reading unrelated files, architecture docs, archive docs, generated outputs, or private data.

## Allowed Scope

- Review current branch diff:
- Optional fixes allowed:
- Severity threshold:

## Forbidden Scope

- Do not review the full repository by default.
- Do not edit files unless fixes are explicitly in scope.
- Do not change schema, `manifest.display`, API/frontend behavior, private artifacts, dependencies, roadmap priority, or docs/archive.

## Stop Conditions

- Stop if the diff implies a public contract change, unclear blast radius, private-data exposure, deployment/trading expansion, or missing approval for fixes.
- Stop if a fix would require files outside the approved scope.

## Checks

- `git status --short --branch`
- `git diff --name-only`
- `git diff --check`
- Narrow tests only if the diff requires them and they are approved.

## Output Format

- Findings
- Open Questions
- Not Changed
- Checks Run
- Scope Check
- Commit Readiness
