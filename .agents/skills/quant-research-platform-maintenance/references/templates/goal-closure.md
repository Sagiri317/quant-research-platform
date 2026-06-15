# Goal Closure Template

Use only inside an approved bounded envelope for long work, overnight work, or review-fix loops.

## Required envelope

```text
Goal:
Branch/worktree:
Allowed files/layers:
Forbidden files/layers:
Subagents allowed: yes/no + roles
Checks required:
Handoff required: yes/no
Stop conditions:
```

## Loop

1. Pre-goal blocker discovery.
2. Slice plan by owner layer.
3. Implement one slice.
4. Run narrow tests.
5. Same issue family scan.
6. Review/fix loop only for in-scope findings.
7. Repeat until definition of done or stop condition.
8. Final truthfulness report.
9. Handoff package if required.

Subagents must get narrow context packets and default to read-only. Main agent owns integration and final truth.
