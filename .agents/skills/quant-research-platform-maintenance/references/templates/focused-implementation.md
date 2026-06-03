# Focused Implementation

## Task Mode

`focused-implementation`

## Recommended Reasoning Effort

Medium / Medium-High

Use Medium for a single module or docs/rules batch. Use Medium-High when the change is contract-adjacent, touches linked surfaces, or has meaningful regression risk.

## Context Budget

- Template loading mode:
  - Expanded Prompt Mode: when the user provides a complete task brief or envelope, do not reload all templates to reconstruct the workflow.
  - Repo Router Mode: when the user gives only a short task intent, start from `auto-task-router.md`, then load this template only if selected.
- Start with `AGENTS.md`, the user request, and the files named by the task.
- Use `rg` to locate current patterns and references before editing.
- Read line ranges first; read full files only when local context is insufficient.
- Report candidate expansion before touching files outside allowed scope.
- Use diff-first review, candidate docs only, targeted checks, and short final reports.

## Allowed Scope

- Files/directories:
- Behaviors/docs to change:
- Narrow tests/checks:

## Forbidden Scope

- Do not change schemas, 0.1 report/API display contract, API/website UI behavior, private artifacts, historical docs, roadmap priority, dependencies, or directory structure unless explicitly approved.
- Do not delete files or run full tests without approval.
- Do not generate smoke/training/evaluation artifacts except under pytest `tmp_path` or an explicitly approved output path.
- Do not commit generated artifacts, read/modify private generated artifacts, or write row-level private data, sample stocks, or full private paths.
- For CLI, artifact writer, exporter, download, ZIP/package, metadata writer, or
  persisted row-level output work, apply the maintenance skill `Private Artifact
  Writer Checklist`.

## Stop Conditions

- Stop if the required change crosses the approved task surface, changes a public contract, needs a new dependency, or has unclear blast radius.
- Stop if generated/private data, deployment/trading, external API, secrets, or credential handling becomes relevant.
- Stop for new framework, new top-level package, new artifact type or required artifact file, A-share adapter capability expansion, model registry, feature store, scheduler, worker, roadmap priority change, or repeated unclear test failure.

## Contract / Safety / Artifact Issue-Family Gate

For contract-adjacent, safety-adjacent, artifact-adjacent, metadata writer,
CLI/API/service, schema inventory, EDA refresh, run discovery, or private-output
boundary work, list the issue-family checklist before implementation. Do not
only fix the file where the bug was observed; scan sibling entrypoints for the
same issue family.

Issue families to consider:

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

Before changing code, include a blast-radius declaration across affected
surfaces, including service, CLI, API, schema/metadata writer, artifact writer,
EDA refresh, run discovery, docs/status, tests, and private-output boundary.

For issue-family fixes, write parameter-family RED tests instead of single-case
smoke tests. Include unsafe identifier variants such as `vendor_factor`,
`vendorFactor`, `factorVendor`, `sourceAlpha`, `localTimestamp`, `../alpha`,
`C:/tmp/alpha`, and `..\alpha`; include allowlist regressions such as
`forward_return_1d`, `walk_forward`, and `random_forest_lite`. Cover duplicate
keys through service, API, and CLI paths when applicable; cover malformed
metadata across status, schema inventory, EDA, and comparison surfaces; cover
optional dependencies in both installed and missing paths.

For high-risk surfaces, assign subagents by issue family rather than generic
review when subagents are allowed: ordering/preflight/duplicate-key,
identifier/path/private-marker sanitizer, artifact freshness / generated
artifact cleanup, API/CLI/service contract, and merge-readiness / test coverage.

Wording constraints: use generic reusable wording. Do not include real commit
SHAs, private paths, row samples, stock-code samples, alpha/performance/
investment claims, or a long historical narrative.

## Branch / Closed-loop Policy

- Branch tasks use `feature/<task-name>`.
- Do not default to `codex/`, `docs/`, or `chore/` branch prefixes.
- Direct `main` is allowed only for explicitly approved low-risk docs-only work; use `direct-main-docs-only.md`.
- Inside an approved closed-loop envelope, proceed through plan, implementation, review/fix, verification, commit, and feature-branch push. Stop only for blocker or escalation items.

## Documentation Impact

- Run `documentation-impact-gate.md` before final reporting.
- Use candidate docs from changed surfaces; do not perform an exhaustive docs scan unless approved by task mode.

## Python / Environment

- Use `<PROJECT_PYTHON>` in reusable prompts and examples.
- Do not hardcode local interpreter paths.
- If a provided platform-specific Python path is needed, use it only for that task.
- If `conda run` fails because of shell or encoding wrapper behavior, a direct project Python invocation may be used with the reason reported.

## Checks

- `git status --short --branch`
- Narrow tests or static checks:
- For optional/default values, omitted values, time windows, folds, splits, ranges, intervals, pagination, limits, fallback behavior, overwrite/resume semantics, or similar boundary logic, include at least one adversarial omitted/default-value test proving validator behavior matches actual runtime behavior.
- `git diff --check`
- `git diff --name-only`

## Output Format

- Changed
- Not Changed
- Checks Run
- Documentation Impact
- Scope Check
- Residual Risks
- Recommended commit message
