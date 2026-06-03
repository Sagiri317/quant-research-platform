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
- For medium+ changes, workflow/template changes, contract-adjacent changes, or active-doc changes, run `documentation-impact-gate.md`.
- Documentation Impact Gate is candidate-based, not a full docs scan.

## Allowed Scope

- Review current branch diff:
- Optional fixes allowed:
- Severity threshold:

## Forbidden Scope

- Do not review the full repository by default.
- Do not edit files unless fixes are explicitly in scope.
- Do not change schema, 0.1 report/API display contract, API/website UI behavior, private artifacts, dependencies, roadmap priority, or historical docs.

## Safety Gates

- Treat review-only prompts as review-only; do not modify files unless the user explicitly approves fixes.
- If the diff touches a CLI writer, artifact writer, exporter, downloader, ZIP/package writer, metadata writer, API/download payload builder, report generator, or persisted row-level output, verify that the maintenance skill `Private Artifact Writer Checklist` was applied or report it as a finding.
- If the diff touches schema, 0.1 report/API display contract, public API, website UI shared contracts, `retired evaluation interface`, or generated artifacts, report the contract-surface stop/approval status before commit readiness.
- Optional/default value semantic alignment: If the diff touches optional/default values, omitted values, time windows, folds, splits, ranges, intervals, pagination, limits, fallback behavior, overwrite/resume semantics, or similar boundary logic, compare parser/config/validator/runtime consumer/metadata/tests together. For each omitted/default value, construct at least one adversarial counterexample and verify validator behavior matches actual runtime behavior. Do not treat validation coverage as sufficient unless the validator and the actual consumer interpret omitted/default values identically.

## Issue-Family Review Gate

For contract/safety/artifact-adjacent diffs, review by issue family rather than
only by changed file. Confirm the diff included a blast-radius declaration
across relevant service, CLI, API, schema/metadata writer, artifact writer, EDA
refresh, run discovery, docs/status, tests, and private-output boundary
surfaces. If a bug is fixed in one entrypoint, scan sibling entrypoints for the
same issue family.

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

For tests, verify parameter-family coverage rather than only smoke cases:
unsafe identifier variants (`vendor_factor`, `vendorFactor`, `factorVendor`,
`sourceAlpha`, `localTimestamp`, `../alpha`, `C:/tmp/alpha`, `..\alpha`),
allowlist regressions (`forward_return_1d`, `walk_forward`,
`random_forest_lite`), duplicate keys across service/API/CLI when applicable,
malformed metadata across status/schema inventory/EDA/comparison, and optional
dependencies in installed and missing paths.

If subagents are used for review, assign read-only subagents by issue family:
ordering/preflight/duplicate-key, identifier/path/private-marker sanitizer,
artifact freshness / generated artifact cleanup, API/CLI/service contract, and
merge-readiness / test coverage.

For pre-merge readiness, answer:

- Did every issue family get checked across all relevant entrypoints?
- Are tests parameterized by variants, not only smoke cases?
- Are API/CLI/service error codes consistent?
- Are disk-loaded metadata paths sanitized?
- Are generated artifacts ignored/untracked?
- Are docs precise without overclaiming?

Wording constraints: durable docs and review reports should use generic
reusable wording. Do not include real commit SHAs, private paths, row samples,
stock-code samples, alpha/performance/investment claims, or a long historical
narrative.

## Verification Honesty

- Do not imply tests passed when they were not run; list skipped pytest/ruff with the reason.

## Stop Conditions

- Stop if the diff implies a public contract change, unclear blast radius, private-data exposure, deployment/trading expansion, or missing approval for fixes.
- Stop if a fix would require files outside the approved scope.

## Checks

- `git status --short --branch`
- `git diff --name-only`
- `git diff --check`
- Documentation Impact Gate when required.
- Narrow tests only if the diff requires them and they are approved.

## Output Format

- Findings
- Open Questions
- Not Changed
- Documentation Impact
- Candidate docs
- Docs checked
- Docs updated
- Docs deferred
- Docs not checked
- Exhaustive docs scan status
- Checks Run
- Scope Check
- Commit Readiness: docs sync not needed because ... / docs sync included / docs sync deferred until after merge / docs sync required before commit
