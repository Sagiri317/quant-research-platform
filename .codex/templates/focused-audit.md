# Focused Audit

## Task Mode

`focused-audit`

## Recommended Reasoning Effort

Medium / High

Use Medium for a narrow diff or single surface. Use High for contract-adjacent, cross-file, private-data boundary, memory/performance, or user-visible workflow risk.

## Context Budget

- Start from the target path, diff, or issue statement.
- Use `rg` for symbols, contracts, and linked docs before full-file reads.
- Read only the target surface, relevant tests, and directly linked contract docs.
- Pre-report any expansion into architecture, archive, generated outputs, private data, or unrelated modules.

## Allowed Scope

- Audit target:
- Risk classes:
- Severity standard:
- Fixes allowed: no, unless explicitly approved.

## Forbidden Scope

- Do not edit files unless the audit envelope allows fixes.
- Do not broaden into full-repo audit by default.
- Do not change schemas, `manifest.display`, API/frontend behavior, private artifacts, deployment/trading scope, dependencies, or roadmap priority.

## Stop Conditions

- Stop if impact crosses multiple architecture layers or blast radius becomes unclear.
- Stop if fixing would require public contract, schema, dependency, directory, archive, private-data, or generated-output changes.

## Checks

- `git status --short --branch`
- `rg "<symbol|contract|term>" <target paths>`
- Targeted static checks only when approved by the task.

## Output Format

- Findings
- Evidence
- Severity
- Files Read
- Not Changed
- Follow-ups
- Scope Check
