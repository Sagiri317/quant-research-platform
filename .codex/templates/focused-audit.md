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
- Use diff-first review and candidate docs only; do not broad-load all templates or docs.

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
- Stop for new dependency/framework, schema or `manifest.display` change, public API/frontend shared contract change, private path/row-level risk, roadmap priority change, trading/deployment scope, or repeated unclear test failure.

## Decision Classification

- No-user-decision items: local findings or fixes fully inside the approved audit/fix envelope.
- User-decision-required / escalation items: choices that affect contracts, scope, semantics, private data, roadmap priority, dependencies, or architecture boundaries.
- Blocked items: issues that cannot proceed because required facts, approvals, clean branch state, or verification are missing.

Each escalation decision must include:

- Decision question
- Options A/B/C
- Recommended option
- Why
- Risk if wrong
- Whether implementation must wait

## Checks

- `git status --short --branch`
- `rg "<symbol|contract|term>" <target paths>`
- Targeted static checks only when approved by the task.

## Output Format

- Findings
- Evidence
- Severity
- No-user-decision items
- User-decision-required / escalation items
- Blocked items
- Files Read
- Not Changed
- Follow-ups
- Scope Check
