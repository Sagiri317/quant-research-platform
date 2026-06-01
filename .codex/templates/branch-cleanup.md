# Branch Cleanup

## Task Mode

`branch-cleanup`

Single-branch cleanup only.

## Allowed Scope

- Delete only the explicitly approved `<BRANCH_NAME>`.
- Verify the branch name exactly before cleanup.
- Verify current branch is `main`, not `<BRANCH_NAME>`.
- Verify `main == origin/main`.
- Before local deletion, confirm `<BRANCH_NAME>` is already contained in `main`.
- Before remote deletion, confirm `origin/<BRANCH_NAME>` is already contained in `origin/main`.
- If the remote branch does not exist, report it and continue without remote deletion.

## Forbidden Scope

- Do not batch-delete merged branches.
- Do not delete multiple branches from inferred patterns.
- Do not delete remote branches unless the user explicitly approved the exact remote branch.
- Do not delete worktrees or files.
- Do not use `git branch -D`.
- Do not force push.
- Do not delete any other branch.

Bulk branch cleanup requires a separate task template, a candidate branch list, and explicit user approval for that list.

## Checks

- `git fetch origin`
- `git status --short --branch`
- `git branch --show-current`
- Verify current branch is `main`.
- Verify `main == origin/main`.
- `git branch --list <BRANCH_NAME>`
- `git merge-base --is-ancestor <BRANCH_NAME> main`
- Remote check only if remote cleanup is explicitly approved.
- `git merge-base --is-ancestor origin/<BRANCH_NAME> origin/main` before deleting a remote branch.

If already-contained status cannot be confirmed, stop and do not delete.

## Output Format

- Branch requested
- Branch deleted
- Remote touched: yes/no
- Checks Run
- Scope Check
