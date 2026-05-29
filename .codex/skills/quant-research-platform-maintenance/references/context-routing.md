# Context Routing

Use this reference to decide which files to read for a QRP task. Keep loading narrow and task-shaped; this file is not an approval layer.

## Context Budget Rules

- Ordinary tasks do not broad-load the repository. Start with `rg` over the requested surface, then read only the matching files.
- Prefer line ranges or focused snippets before full files when the file is large and the relevant section is known.
- Load future architecture, roadmap, archive, generated evidence, private data, or broad code maps only when the task mode or approved envelope requires them.
- `pre-commit-review` is diff-only by default: read `git status`, `git diff`, and only the files needed to understand changed lines.
- Subagents read only their assigned files, assigned directories, or assigned diff slices. They do not independently expand to full-repo context.
- If the likely file budget needs to expand beyond the task-specific set, report the candidate expansion first with the reason and wait when expansion would touch a stop condition.
- Goal / Overnight mode may use broader context only inside an approved envelope with a context loading plan, phase plan, subagent plan, verification cadence, and stop conditions.
- Prefer checked-in routing and templates under `.codex/templates/` over ad hoc context packs.

| Task | Load |
| --- | --- |
| Any medium+ task | `AGENTS.md`, task brief or user request, and only the linked files below that match the task surface. |
| Goal / Overnight Run | `AGENTS.md`, `.codex/skills/quant-research-platform-maintenance/SKILL.md`, this context routing file, task-specific code map, task-specific contract reference, and relevant tests. |
| Subagent | Parent-approved envelope, assigned files/directories only, task-specific contract docs, and relevant tests/checks only. |
| Documentation Sync Gate | `AGENTS.md`, `.codex/skills/quant-research-platform-maintenance/SKILL.md`, changed-file-specific code map, and relevant active docs only. |
| Archive Candidate Check | `docs/archive/README.md`, current active doc containing the candidate, and `PROJECT_STATUS.md` only if status/handoff changes. |
| Artifact schema, `manifest.display`, API/download, frontend evidence, generated runs | `.codex/skills/quant-research-platform-maintenance/references/contracts-and-verification.md`; add `docs/architecture/engineering-standards.md` when module boundaries, validation, generated outputs, API/frontend behavior, or security boundaries are part of the task. |
| Diagnostics semantics, factor evaluation meaning, tradability, costs, liquidity, capacity, benchmarks, risk, lineage, or trading-adjacent wording | `docs/architecture/quant-research-standards.md` plus the contracts reference when artifacts/API/frontend are touched. |
| Code map edits | `.codex/skills/quant-research-platform-maintenance/references/code-map-flow.md` and the specific `docs/code-map/` files being changed. |
| README, architecture index, project-facing docs | The target doc, `AGENTS.md`, and only directly linked current docs needed to verify wording. |
| Decision task, architecture planning, or future capability boundary | `docs/architecture/target-architecture-and-capabilities.md`, `docs/architecture/next-stage-roadmap.md`, and the relevant standards. Ordinary coding tasks should not load target architecture by default. |
| Worktree, branch, or operations planning | `docs/architecture/next-stage-worktree-plan.md` while it remains active; if archived later, load it only as historical operations context. |
| Historical plans/specs under `docs/archive/superpowers/` | Load only when tracing prior decisions, negative/rejection coverage, or explicitly cited historical context. Do not treat these files as current contract truth without confirmation from current docs, code, or tests. |

Ordinary coding tasks do not load `docs/architecture/target-architecture-and-capabilities.md` by default. Load target architecture only for future capability, architecture planning, or phase-boundary tasks.

Subagents do not read the full project by default. They read only the narrow context packet specified by the parent-approved envelope.
