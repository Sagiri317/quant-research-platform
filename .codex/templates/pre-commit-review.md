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
- Do not change schema, `manifest.display`, API/frontend behavior, private artifacts, dependencies, roadmap priority, or docs/archive.

## Safety Gates

- Treat review-only prompts as review-only; do not modify files unless the user explicitly approves fixes.
- If the diff touches a CLI writer, artifact writer, exporter, downloader, ZIP/package writer, metadata writer, API/download payload builder, report generator, or persisted row-level output, verify that the maintenance skill `Private Artifact Writer Checklist` was applied or report it as a finding.
- If the diff touches schema, `manifest.display`, public API, frontend shared contracts, `EVALUATION_INTERFACE`, or generated artifacts, report the contract-surface stop/approval status before commit readiness.

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
