# Context Routing

Use this reference to decide which files to read for a QRP task. Keep loading narrow and task-shaped; this file is not an approval layer.

## Context Budget Rules

- Ordinary tasks do not broad-load the repository. Start with `rg` over the requested surface, then read only the matching files.
- Prefer line ranges or focused snippets before full files when the file is large and the relevant section is known.
- Load future architecture, roadmap, archive, generated evidence, private data, or broad code maps only when the task mode or approved envelope requires them.
- `pre-commit-review` is diff-only by default: read `git status`, `git diff`, and only the files needed to understand changed lines.
- `pre-commit-review` uses `documentation-impact-gate.md` for medium+ changes, workflow/template changes, contract-adjacent changes, or active-doc changes.
- Subagents read only their assigned files, assigned directories, or assigned diff slices. They do not independently expand to full-repo context.
- If the likely file budget needs to expand beyond the task-specific set, report the candidate expansion first with the reason and wait when expansion would touch a stop condition.
- Goal / Overnight mode may use broader context only inside an approved envelope with a context loading plan, phase plan, subagent plan, verification cadence, and stop conditions.
- Prefer checked-in routing and templates under `.agents/skills/quant-research-platform-maintenance/references/templates/` over ad hoc context packs.
- Expanded Prompt Mode: when the user provides a complete task brief or approved envelope, follow that prompt and load only named or directly relevant files.
- Repo Router Mode: when the user gives only a short task intent, start with `.agents/skills/quant-research-platform-maintenance/references/templates/auto-task-router.md`, then load the smallest selected template set.
- Do not full-load all templates, all docs, generated outputs, private data, or archive docs by default.

| Task | Load |
| --- | --- |
| Short task intent with no expanded prompt | `.agents/skills/quant-research-platform-maintenance/references/templates/auto-task-router.md`, then the smallest selected template set. |
| Expanded prompt or approved envelope | The provided prompt/envelope plus only named or directly relevant files; do not reload all templates to reconstruct the workflow. |
| Any medium+ task | `AGENTS.md`, task brief or user request, and only the linked files below that match the task surface. |
| Ordinary QRP current-context primer | `.agents/briefs/current-qrp-status.md` before broad-loading `PROJECT_STATUS.md`, `README.md`, or the roadmap; this primer is not a `PROJECT_STATUS.md` replacement. |
| Goal / Overnight Run | `AGENTS.md`, `.agents/skills/quant-research-platform-maintenance/SKILL.md`, this context routing file, task-specific code map, task-specific contract reference, and relevant tests. |
| Subagent | Parent-approved envelope, assigned files/directories only, task-specific contract docs, and relevant tests/checks only. |
| Documentation Impact Gate | `.agents/skills/quant-research-platform-maintenance/references/templates/documentation-impact-gate.md`, changed-file-specific candidates, and relevant active docs only. |
| Candidate docs sync check | `documentation-impact-gate.md`, changed-file-specific code map, and relevant active docs only; use candidate docs and `rg` first. |
| Post-merge documentation sync | `.agents/skills/quant-research-platform-maintenance/references/templates/post-merge-documentation-sync.md`, `.agents/skills/quant-research-platform-maintenance/references/templates/documentation-impact-gate.md`, merge changed-file list, and candidate active docs only. |
| Dependency or contract approval | `.agents/skills/quant-research-platform-maintenance/references/templates/dependency-contract-approval.md` plus only the directly affected config, contract, template, or code references needed to classify approval. |
| Archive Candidate Check | `docs/archive/README.md`, current active doc containing the candidate, and `PROJECT_STATUS.md` only if status/handoff changes. |
| Artifact schema, `manifest.display`, API/download, frontend evidence, generated runs | `.agents/skills/quant-research-platform-maintenance/references/contracts-and-verification.md`; add `docs/architecture/engineering-standards.md` when module boundaries, validation, generated outputs, API/frontend behavior, or security boundaries are part of the task. |
| Diagnostics semantics, factor evaluation meaning, tradability, costs, liquidity, capacity, benchmarks, risk, lineage, or trading-adjacent wording | `docs/architecture/quant-research-standards.md` plus the contracts reference when artifacts/API/frontend are touched. |
| Code map edits | `.agents/skills/quant-research-platform-maintenance/references/code-map-flow.md` and the specific `docs/code-map/` files being changed. |
| README, architecture index, project-facing docs | The target doc, `AGENTS.md`, and only directly linked current docs needed to verify wording. |
| CLI/artifact writer/exporter/download/ZIP/metadata writer or persisted row-level output task | `.agents/skills/quant-research-platform-maintenance/SKILL.md` `Private Artifact Writer Checklist`; add the contracts reference only when schema, `manifest.display`, API/download, frontend, generated-run, or public boundary behavior may change. |
| Decision task, architecture planning, or future capability boundary | `docs/architecture/target-architecture-and-capabilities.md`, `docs/architecture/next-stage-roadmap.md`, and the relevant standards. Ordinary coding tasks should not load target architecture by default. |
| Governance, architecture, standards, Codex workflow redesign, task-mode policy, or documentation topology task | Start with `AGENTS.md`, this routing file, and the relevant task template. Use `rg` to locate narrow sections in `docs/architecture/ai-native-engineering-standard.md` only when that standard is directly named, being edited, or needed to resolve a governance conflict; do not full-load it by default. |
| Branch, commit, push, fast-forward merge, or branch cleanup task | Use `.agents/skills/quant-research-platform-maintenance/references/templates/commit-and-push.md`, `.agents/skills/quant-research-platform-maintenance/references/templates/branch-push.md`, `.agents/skills/quant-research-platform-maintenance/references/templates/ff-merge-and-push.md`, or `.agents/skills/quant-research-platform-maintenance/references/templates/branch-cleanup.md` as the current execution template. Branch cleanup is single-branch only and must confirm merge-base containment before deletion. Do not load archived worktree plans by default. |
| Goal / overnight execution planning | Use `.agents/skills/quant-research-platform-maintenance/references/templates/goal-overnight.md` plus `.agents/skills/quant-research-platform-maintenance/references/templates/pre-goal-blocker-discovery.md` and the approved envelope. Preserve the context budget rules above. |
| Historical worktree or operations planning | Load `docs/archive/operations/next-stage-worktree-plan-2026-05.md` only when tracing historical branch/worktree decisions. It is not current Codex task-rule truth. |
| Historical plans/specs under `docs/archive/superpowers/` | Load only when tracing prior decisions, negative/rejection coverage, or explicitly cited historical context. Do not treat these files as current contract truth without confirmation from current docs, code, or tests. |

Ordinary coding tasks do not load `docs/architecture/target-architecture-and-capabilities.md` by default. Load target architecture only for future capability, architecture planning, or phase-boundary tasks.

Ordinary coding tasks do not load `docs/architecture/ai-native-engineering-standard.md` by default. Prefer targeted `rg` matches and focused line ranges; load the full standard only when the user explicitly asks to review or rewrite that standard, or when narrow reads cannot resolve a governance conflict.

Subagents do not read the full project by default. They read only the narrow context packet specified by the parent-approved envelope.
