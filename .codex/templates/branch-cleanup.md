# Branch Cleanup

## Task Mode

`branch-cleanup`

Single-branch cleanup only.

## Allowed Scope

- Delete only the explicitly approved `<BRANCH_NAME>`.
- Verify the branch name exactly before cleanup.
- Verify current branch is not `<BRANCH_NAME>`.

## Forbidden Scope

- Do not batch-delete merged branches.
- Do not delete multiple branches from inferred patterns.
- Do not delete remote branches unless the user explicitly approved the exact remote branch.
- Do not delete worktrees or files.

Bulk branch cleanup requires a separate task template, a candidate branch list, and explicit user approval for that list.

## Checks

- `git status --short --branch`
- `git branch --show-current`
- `git branch --list <BRANCH_NAME>`
- Remote check only if remote cleanup is explicitly approved.

## Output Format

- Branch requested
- Branch deleted
- Remote touched: yes/no
- Checks Run
- Scope Check
