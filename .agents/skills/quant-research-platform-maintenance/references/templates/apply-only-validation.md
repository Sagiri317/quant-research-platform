# Apply-only Validation Template

Use when the user supplies a patch/zip or external design and says Codex should apply exactly.

## Rules

- Do not redesign.
- Do not substitute alternative architecture.
- Apply the provided patch/files exactly unless there is a mechanical conflict.
- If conflict occurs, report the conflict and stop unless the apply prompt authorizes trivial path/line resolution.
- Validate only.
- Do not modify business code unless the patch itself explicitly contains those changes and the user approved them.

## Checks

```text
git status --short --branch
git diff --name-only
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only
git diff --check
<markdown/path checks if available>
```

## Output

```text
Applied files:
Conflicts:
Validation run:
Validation not run:
Business code touched: yes/no
Public API touched: yes/no
Private data risk: yes/no
Residual risks:
```
