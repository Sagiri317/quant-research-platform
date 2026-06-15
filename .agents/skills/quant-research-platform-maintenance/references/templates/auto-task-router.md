# Auto Task Router

Use only when the user gives a short prompt or no complete execution envelope.

## Steps

1. Classify task mode using `../task-modes.md`.
2. Identify branch safety: main / feature branch / unknown.
3. Identify whether the task is read-only, docs-only, implementation, governance, handoff, merge, release, or apply-only.
4. Load one matching template, not all templates.
5. If any stop condition appears, ask before editing.

## Output before work

```text
Task mode:
Template:
Branch plan:
Allowed scope:
Forbidden scope:
Context to load:
Stop conditions:
```

## Ambiguity defaults

- Review-like wording => read-only review.
- Handoff/package wording => handoff packaging, not implementation.
- Merge/push wording => git operations gate, not code changes.
- Governance wording => governance-only; do not edit `src/**` or business tests.
- If still unclear => stop and ask.
