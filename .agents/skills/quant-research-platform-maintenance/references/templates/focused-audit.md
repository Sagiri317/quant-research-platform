# Focused Audit Template

Use for independent review, pre-commit review, architecture review, or review-only subagent work.

## Rules

- Default read-only. Do not edit unless the user explicitly converts review into fix mode.
- Review by issue family: contract, public surface, private safety, artifact truth, owner layer, naming, performance/memory, docs/status, script portability.
- Cite exact files/lines or commands when possible.
- Separate confirmed findings from hypotheses and follow-up candidates.

## Output

```text
Scope reviewed:
Files read:
Findings:
Evidence:
Severity:
Owner layer:
Suggested fix:
Out-of-scope candidates:
Checks not run:
```
