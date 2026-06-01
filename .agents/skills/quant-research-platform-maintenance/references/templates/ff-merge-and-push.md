# Ff Merge And Push

## Task Mode

`ff-merge-and-push`

## Recommended Reasoning Effort

Low

## Context Budget

- Inspect approved feature branch, `main`, merge-base, and status only.
- Do not broad-load source files unless fast-forward safety is unclear.

## Allowed Scope

- Approved feature branch:
- Target branch: `main`
- Merge type: fast-forward only.
- Push target: `main` after merge.

## Forbidden Scope

- No merge commit, squash, rebase, amend, force push, or conflict resolution unless explicitly approved.
- Do not merge if `main` has local uncommitted changes or if fast-forward is impossible.

## Stop Conditions

- Stop if the feature branch is not approved.
- Stop if fast-forward is impossible.
- Stop if local or remote `main` status is unclear.
- Stop if checks required before merge are missing or failed.
- Stop if the expected feature head SHA or base / merge-base was provided and does not match.

## Checks

- `git status --short --branch`
- `git fetch origin`
- Verify feature head if expected SHA is provided.
- Verify base / merge-base if provided.
- Verify `main` equals `origin/main` before merge.
- `git merge --ff-only <feature-branch>`
- Push `main`.
- Capture the merged commit SHA for the transient final report.
- Verify `HEAD` equals `origin/main` after push.
- `git status --short --branch` after push.
- Do not delete the feature branch unless explicitly approved.

## Output Format

- Merged branch
- Merged commit SHA
- Pushed branch: `main`
- Main status
- Checks Run
- Scope Check
