# Git Operations Template

Use for commit, push, merge gate, release gate, or branch cleanup. Do not combine with business implementation unless the user explicitly approved a closed-loop envelope.

## Common checks

```text
git status --short --branch
git branch --show-current
git log --oneline --decorate -5
git diff --name-only
```

## Commit/push rules

- Confirm changed files are in scope.
- Do not commit unrelated changes.
- Do not push `main` unless explicitly approved.
- No amend/rebase/squash/force-push unless explicitly approved.

## Merge gate rules

- Fetch remote.
- Verify source branch and target branch exactly.
- Verify target branch cleanliness.
- Prefer fast-forward merge when user requested it.
- Do not resolve merge conflicts by redesigning code unless explicitly approved.

## Branch cleanup rules

- Delete only explicitly named branches.
- Verify branch is merged before deletion.
- No batch deletion or pattern deletion.
- No `git branch -D` unless explicitly approved.

## Output

```text
Operation:
Branch/source/target:
Changed files:
Checks run:
Checks not run:
Result:
```
