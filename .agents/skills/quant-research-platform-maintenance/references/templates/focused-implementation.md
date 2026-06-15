# Focused Implementation Template

## Pre-edit

- Task mode: `standard-implementation` or approved small slice of `goal-closure`.
- Read `AGENTS.md`, current status, user principles if architecture/naming/public surface is touched.
- Declare owner layer and expected changed files.
- Identify public/private/artifact impacts.
- Identify red test or existing failing guard. If no test is appropriate, explain why.

## Implementation rules

- Modify owner layer first.
- Do not add public API/CLI/registry/store/facade/compatibility shim unless approved.
- Do not fix unrelated P2/P3 items; route them to owner layer follow-up.
- Keep names semantic and stable.
- Keep private-derived outputs under approved private roots.

## Verification

```text
git status --short --branch
git diff --name-only
git diff --check
<targeted pytest for touched owner layer>
<ruff if Python files changed>
<same issue family scan>
```

## Final report

```text
Changed:
Not changed:
Owner layer:
Public surface:
Private safety:
Tests/checks run:
Checks not run:
Docs/status sync:
Residual risks:
```
