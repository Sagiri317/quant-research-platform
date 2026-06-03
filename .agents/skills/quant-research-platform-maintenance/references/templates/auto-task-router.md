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
- Post-merge doc or handoff sync: use `post-merge-documentation-sync.md`.
- Candidate-based docs impact check: use `documentation-impact-gate.md`.
- Dependency, schema, 0.1 report/API display contract, public API, website UI shared contract, runtime boundary, or other contract approval: use `dependency-contract-approval.md`.
- Standard implementation: use `focused-implementation.md`.
- Focused review/audit: use `focused-audit.md` or `pre-commit-review.md`.
- Goal, overnight, or review-and-fix loop: use `goal-overnight.md`; run `pre-goal-blocker-discovery.md` before implementation.
- Commit existing changes: use `commit-and-push.md`.
- Push existing feature branch commits: use `branch-push.md`.
- Fast-forward merge to `main`: use `ff-merge-and-push.md`.
- Delete one approved branch: use `branch-cleanup.md`.

## One-line Prompt Routing Gate

Short prompts route through `auto-task-router` first unless the user provides a complete approved envelope.

For any one-line, vague, or underspecified user prompt, Codex must classify the task before editing files, committing, pushing, merging, or changing contracts.

Classify as one of: docs-only, docs/rules, review-only, implementation, commit/push, merge/push, test/verification-only, artifact/export/download/package/report generation, contract-surface change, or ambiguous / needs clarification.

Then select the minimal relevant template:

- docs sync -> `docs-sync.md` or `post-merge-documentation-sync.md`.
- implementation -> `focused-implementation.md`.
- review -> `focused-audit.md` or `pre-commit-review.md`.
- commit/push -> `commit-and-push.md` or `branch-push.md`.
- fast-forward merge -> `ff-merge-and-push.md`.
- test-only -> verification path without file edits.
- artifact/export/API/schema/report tasks -> route through the relevant implementation template and required safety gates, including the maintenance skill `Private Artifact Writer Checklist` when persisted artifacts, downloads, packages, reports, metadata, or row-level outputs are involved.

Do not treat a one-line task approval as approval to commit directly to `main`, push `main`, merge branches, change schema/API/website UI/contracts, write private-derived artifacts to ordinary/public output, or convert review-only into implementation.

If routing is ambiguous, stop and ask or choose the safer non-main branch path.

## Branch Decision

- Use `feature/<task-name>` for tasks that need a branch.
- Do not default to `codex/`, `docs/`, or `chore/` branch prefixes.
- Low-risk docs-only work may stay on `main` only when the user explicitly approved direct-main docs-only execution.
- High-impact docs/rules work still uses `feature/<task-name>`.

## Closed-loop Eligibility

- Expanded Prompt Mode: if the user provides a complete approved envelope, follow that envelope.
- Repo Router Mode: if the user gives a short intent and explicitly says closed-loop approved, Codex may construct the smallest safe envelope from the selected template.

If closed-loop is approved, proceed only when the task stays inside that bounded envelope and no blocker or escalation item appears. Then continue through plan, implementation, review/fix, verification, commit, and feature-branch push as applicable.

## Stop Conditions

- Working tree dirty or base branch unclear.
- New dependency, framework, top-level package, scheduler, worker, model registry, feature store, deployment, external API, broker, exchange connector, paper/live trading, or trading-adjacent capability.
- Schema, 0.1 report/API display contract, public API, website UI shared contract, artifact type, required artifact file, A-share adapter capability, private data/path, roadmap priority, directory structure, or unclear blast radius.
- Repeated verification failure with unclear cause.

## Output Format

- Selected template
- Template loading mode
- Branch decision
- Closed-loop eligibility
- Stop conditions hit
