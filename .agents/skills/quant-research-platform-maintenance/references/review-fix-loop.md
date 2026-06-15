# Review-Fix Loop

Use this reference for independent review, goal closure, or review-after-fix
work.

## Loop

1. Record review scope and issue families.
2. Review by issue family, not only by changed file.
3. For each finding, record severity, evidence, owner layer, and required fix.
4. Fix only in approved scope.
5. Rerun the narrow check or rereview that proves the finding closed.
6. Keep unresolved issues as open; do not hide them in a PASS summary.

## Issue Families

Use families relevant to the diff, for example:

- public surface expansion;
- private-output boundary;
- artifact truth or stale artifact;
- metadata/status truthfulness;
- optional/default semantics;
- owner-layer leakage;
- script portability;
- docs/status drift;
- performance or memory regression risk.

## Subagent Use

Subagents are read-only unless explicitly approved. Assign them by issue family
and require concise evidence-backed findings. The main agent integrates results,
chooses fixes, reruns checks, and owns final claims.
