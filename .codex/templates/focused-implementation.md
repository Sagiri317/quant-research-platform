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

- Do not change schemas, `manifest.display`, API/frontend behavior, private artifacts, docs/archive, roadmap priority, dependencies, or directory structure unless explicitly approved.
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
