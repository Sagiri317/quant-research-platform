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

## Boundary Logic Gate

- Optional/default value semantic alignment: If the diff touches optional/default values, omitted values, time windows, folds, splits, ranges, intervals, pagination, limits, fallback behavior, overwrite/resume semantics, or similar boundary logic, compare parser/config/validator/runtime consumer/metadata/tests together. For each omitted/default value, construct at least one adversarial counterexample and verify validator behavior matches actual runtime behavior. Do not treat validation coverage as sufficient unless the validator and the actual consumer interpret omitted/default values identically.

## Issue-Family Audit Gate

For contract/safety/artifact-adjacent audits, list the issue-family checklist
before reporting findings. Do not stop at the file where a symptom appears; scan
sibling service, CLI, API, schema/metadata writer, artifact writer, EDA refresh,
run discovery, docs/status, tests, and private-output boundary surfaces for the
same family. Require a blast-radius declaration when the audit turns into an
implementation or review-and-fix task.

Issue families to check:

- duplicate-before-input-read / duplicate-before-row-level-write
- preflight-before-row-read
- unsafe identifier before metadata/API/CLI echo
- sensitive text and private-marker handling in metadata keys and values
- schema inventory redaction for private/vendor/source/local markers
- stale generated artifact refresh / cleanup
- disk-loaded metadata numeric/string sanitization
- CLI/API/service failure-code consistency
- configured root / run discovery consistency
- long/wide artifact consistency
- deterministic seed / optional dependency behavior
- private input + public output boundary
- row-level private-derived artifact tracking

Audit test quality by parameter family, not single examples: unsafe identifier
variants (`vendor_factor`, `vendorFactor`, `factorVendor`, `sourceAlpha`,
`localTimestamp`, `../alpha`, `C:/tmp/alpha`, `..\alpha`), allowlist regressions
(`forward_return_1d`, `walk_forward`, `random_forest_lite`), duplicate keys
across service/API/CLI when applicable, malformed metadata across status/schema
inventory/EDA/comparison, and optional dependencies in installed and missing
paths.

If review subagents are allowed, assign them by issue family and keep them
read-only: ordering/preflight/duplicate-key, identifier/path/private-marker
sanitizer, artifact freshness / generated artifact cleanup, API/CLI/service
contract, and merge-readiness / test coverage.

Wording constraints: use generic reusable wording. Do not include real commit
SHAs, private paths, row samples, stock-code samples, alpha/performance/
investment claims, or a long historical narrative.

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
