# Handoff Packaging Template

Use for review bundle, handoff zip/json, git bundle inclusion, or public-safe evidence package generation.

## Rules

- Use the repo/local `.run` official handoff tooling when available.
- If `.run` tooling is missing, report `BLOCKED` or `NOT_RUN`; do not invent PASS.
- Include implementation summary, decisions, evidence, tests/smoke/compare results, performance notes, open issues, and review artifacts as required by the task.
- Include a git bundle only when the handoff rules require it.
- Exclude private data and blocked file types.
- Old handoff zip/json is not a source of truth unless explicitly named.

## Safety checks

```text
python .agents/skills/quant-research-platform-maintenance/scripts/package_safety_check.py <zip-or-dir>
<repo .run verify_handoff_package.py command, if available>
<repo .run handoff_safety.py command, if available>
```

## Output

```text
Package path:
Manifest path:
Git bundle included: yes/no/not applicable
Package safety: PASS/FAIL/NOT_RUN/BLOCKED
Verification: PASS/FAIL/NOT_RUN/BLOCKED
Excluded private data: yes/no
Checks not run:
Residual risks:
```
