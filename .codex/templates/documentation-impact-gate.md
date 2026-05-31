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

- `.codex/templates/*`: check related templates, maintenance skill, and context routing.
- `AGENTS.md`: check hard safety boundary references only.
- `.codex/skills/*/SKILL.md`: check directly linked routing/reference files only.
- Contract docs: check only the active docs mapped by the maintenance skill.
- Status docs: check only if the task changes current topic, phase, milestone, blocker, next step, public readiness, or data/API/Git/Conda/privacy boundary.

## Output Format

- Changed surface
- Candidate docs
- Docs checked
- Docs updated
- Docs deferred
- Docs not checked
- Exhaustive docs scan status
