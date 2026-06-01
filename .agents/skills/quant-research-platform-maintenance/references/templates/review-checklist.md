# Review Checklist

General review checklist for QRP docs, rules, contracts, and implementation surfaces. For pre-commit branch review, use `pre-commit-review.md` first; this checklist can be a supporting lens when the diff indicates broader risk.

## Task Mode

`review-checklist`

## Recommended Reasoning Effort

Medium

## Context Budget

- Use as a checklist after selecting a task-specific template.
- Read only the diff, target files, and linked contracts needed by the selected template.
- Do not broaden to full-repo review unless the approved task mode calls for it.

## Allowed Scope

- Review surface:
- Checklist lenses:
- Fixes allowed:

## Forbidden Scope

- Do not change schemas, `manifest.display`, API/frontend behavior, private artifacts, docs/archive, dependencies, deployment/trading scope, or roadmap priority unless explicitly approved.
- Do not delete files or run full tests without approval.

## Stop Conditions

- Stop if the review finds public contract drift, unclear schema semantics, unclear blast radius, private-data exposure risk, or a required fix outside approved scope.

## Checks

- `git status --short --branch`
- `git diff --name-only`
- `git diff --check`
- Narrow task-specific checks:

## Output Format

- Findings
- Open Questions
- Checks Run
- Scope Check

- [ ] Contract drift
- [ ] Schema drift
- [ ] Semantic drift
- [ ] Time semantics drift
- [ ] Phase drift
- [ ] Scope creep
- [ ] Abstraction creep
- [ ] Future capability contamination
- [ ] Half migration
- [ ] Hidden dependency
- [ ] Hardcoded path
- [ ] Evidence flow bypass
- [ ] Missing tests
- [ ] Missing review summary
