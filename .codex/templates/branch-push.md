# Branch Push

## Task Mode

`branch-push`

## Recommended Reasoning Effort

Low

## Context Budget

- Inspect current branch, upstream, and unpushed commits only.
- Do not read source files unless branch identity or commit contents are unclear.

## Allowed Scope

- Existing local commit(s) on feature branch:
- Remote:
- Push target:

## Forbidden Scope

- No merge, rebase, amend, force push, squash, branch deletion, or dependency changes unless explicitly approved.
- Do not push `main` through this template.

## Stop Conditions

- Stop if current branch is `main`.
- Stop if working tree is dirty and the dirty changes are unrelated or unclear.
- Stop if upstream/remote target is ambiguous.

## Checks

- `git status --short --branch`
- `git branch --show-current`
- `git log --oneline --decorate -5`
- `git status --short --branch` after push

## Output Format

- Pushed branch
- Commit SHA(s)
- Remote
- Checks Run
- Scope Check
