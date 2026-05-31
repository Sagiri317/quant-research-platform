# Pre Goal Blocker Discovery

## Task Mode

`pre-goal-blocker-discovery`

Run before implementation for `goal-overnight` or other long closed-loop work.

## Inputs

- Approved envelope.
- Current branch/worktree state.
- Allowed and forbidden scope.
- Stop conditions.
- Candidate files and checks.

## Discovery Checks

- Confirm branch, base, and working tree state.
- Confirm allowed path scope.
- Identify likely changed files before editing.
- Identify likely checks before editing.
- Search for obvious blockers with `rg` over the target surface only.
- Do not read private data, generated evidence, archive docs, or unrelated modules unless named in the envelope.

## Decision

- Can proceed: yes/no
- Blockers:
- Escalations:
- Safe implementation slice:
- Files likely to change:
- Tests likely to run:

If `Can proceed` is `no`, do not edit files.

## Stop Conditions

Stop for any blocker or escalation listed by the approved envelope, including schema, `manifest.display`, public API/frontend shared contract, new dependency/framework, directory restructuring, private/generated data edits, trading/deployment scope, unclear blast radius, or repeated unclear test failure.
