# Auto Task Router

## Task Mode

`auto-task-router`

Use this only in Repo Router Mode: the user gives a short task intent and has not provided a full expanded task prompt.

## Template Loading Mode

- Expanded Prompt Mode: if the user provides a complete task brief or approved envelope, do not reload all templates to rebuild the workflow. Follow the prompt and load only named or directly relevant files.
- Repo Router Mode: read this router first, then load the smallest matching template set below.
- Do not full-scan the repository, all docs, all templates, generated outputs, or private data by default.

## Routing

- Inspect-only: use `discovery-only.md`.
- Narrow active-doc sync: use `docs-sync.md`.
- Explicit direct-main docs-only task: use `direct-main-docs-only.md`.
- Standard implementation: use `focused-implementation.md`.
- Focused review/audit: use `focused-audit.md` or `pre-commit-review.md`.
- Goal, overnight, or review-and-fix loop: use `goal-overnight.md`; run `pre-goal-blocker-discovery.md` before implementation.
- Commit existing changes: use `commit-and-push.md`.
- Push existing feature branch commits: use `branch-push.md`.
- Fast-forward merge to `main`: use `ff-merge-and-push.md`.
- Delete one approved branch: use `branch-cleanup.md`.

## Branch Decision

- Use `feature/<task-name>` for tasks that need a branch.
- Do not default to `codex/`, `docs/`, or `chore/` branch prefixes.
- Low-risk docs-only work may stay on `main` only when the user explicitly approved direct-main docs-only execution.
- High-impact docs/rules work still uses `feature/<task-name>`.

## Closed-loop Eligibility

Closed-loop is allowed only when the user approved an envelope that defines goal, allowed scope, forbidden scope, stop conditions, verification, commit/push policy, and final report format.

If closed-loop is approved, proceed through plan, implementation, review/fix, verification, commit, and feature-branch push. Stop only for blocker or escalation items.

## Stop Conditions

- Working tree dirty or base branch unclear.
- New dependency, framework, top-level package, scheduler, worker, model registry, feature store, deployment, external API, broker, exchange connector, paper/live trading, or trading-adjacent capability.
- Schema, `manifest.display`, public API, frontend shared contract, artifact type, required artifact file, A-share adapter capability, private data/path, roadmap priority, directory structure, or unclear blast radius.
- Repeated verification failure with unclear cause.

## Output Format

- Selected template
- Template loading mode
- Branch decision
- Closed-loop eligibility
- Stop conditions hit
