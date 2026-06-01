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
- Do not merge `main`.

## Stop Conditions

- Stop if unrelated or suspicious changes are present.
- Stop if current branch is `main` and the user did not approve committing on `main`.
- Stop if tests/checks required by the task have not been run or their status is unclear.
- Stop if the Documentation Impact Gate result is missing for medium+ or workflow-rule changes.
- Stop if changed paths exceed the approved scope.

## Checks

- `git status --short --branch`
- `git diff --name-only`
- Verify allowed path scope.
- Verify Documentation Impact Gate result exists when required.
- `git diff --check`
- Task-specific checks already run:
- Confirm the commit exists without printing SHA values.
- Push current branch.
- `git status --short --branch` after push

## Output Format

- Changed
- Commit status, without printing SHA values
- Pushed branch
- Checks Run
- Not Changed
- Documentation Impact
- Scope Check
