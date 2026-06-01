# Documentation Impact Gate

## Task Mode

`documentation-impact-gate`

Use before final reporting for medium+ tasks and any change to workflow rules, contracts, templates, skills, or active docs.

## Candidate-based Gate

- This is candidate-based, not an exhaustive docs scan.
- Choose candidate docs from changed surfaces and direct references.
- Use `rg` first, then read only matching snippets or directly relevant sections.
- Do not full-load all Markdown files by default.
- Prefer `deferred` or `not checked` notes over broad context loading.
- Full docs scan is allowed only for full docs sync, docs topology cleanup, or template sync audit tasks.

## Candidate Selection

- `.agents/skills/quant-research-platform-maintenance/references/templates/*`: check related templates, maintenance skill, and context routing.
- `AGENTS.md`: check hard safety boundary references only.
- `.agents/skills/*/SKILL.md`: check directly linked routing/reference files only.
- Contract docs: check only the active docs mapped by the maintenance skill.
- Status docs: check only if the task changes current topic, phase, milestone, blocker, next step, public readiness, or data/API/Git/Conda/privacy boundary.

## Active Docs Bloat Gate

- For touched active docs, report before/after line count.
- If a touched active doc grows by more than 20 lines or more than 15%, either compact an older section in the same commit or explicitly justify the growth in the final report.
- For status / entry / roadmap docs, replace stale facts instead of appending history.
- Do not turn active docs into Git logs.
- Do not store real commit SHA in active docs.
- Prefer links to authoritative docs over duplicated long explanations.
- This gate applies only to touched docs. It is a prompt / review gate, not a hard failure or exhaustive docs scan.

## Output Format

- Changed surface
- Candidate docs
- Docs checked
- Docs updated
- Docs deferred
- Docs not checked
- Exhaustive docs scan status
- Before/after line counts for touched active docs
