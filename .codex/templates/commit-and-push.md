# Commit And Push

## Task Mode

`commit-and-push`

## Recommended Reasoning Effort

Low

## Context Budget

- Inspect current branch status and current local changes only.
- Review `git diff --name-only`, staged diff, and unstaged diff enough to confirm scope.
- Do not broad-load repo or run unrelated tests.

## Allowed Scope

- Confirmed local changes on current branch:
- Commit message:
- Push target: current branch.

## Forbidden Scope

- No amend, rebase, force push, squash, merge, or branch switch unless explicitly approved.
- Do not include unrelated changes, secrets, private artifacts, generated private outputs, or files outside confirmed scope.
- Do not delete files or change dependencies.

## Stop Conditions

- Stop if unrelated or suspicious changes are present.
- Stop if current branch is `main` and the user did not approve committing on `main`.
- Stop if tests/checks required by the task have not been run or their status is unclear.

## Checks

- `git status --short --branch`
- `git diff --name-only`
- `git diff --check`
- Task-specific checks already run:
- `git log -1 --format=%H` after commit
- `git status --short --branch` after push

## Output Format

- Changed
- Commit SHA: full SHA
- Pushed branch
- Checks Run
- Not Changed
- Scope Check
