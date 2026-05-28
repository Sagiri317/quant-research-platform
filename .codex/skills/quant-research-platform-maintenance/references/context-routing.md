# Context Routing

Use this reference to decide which files to read for a QRP task. Keep loading narrow and task-shaped; this file is not an approval layer.

| Task | Load |
| --- | --- |
| Any medium+ task | `AGENTS.md`, task brief or user request, and only the linked files below that match the task surface. |
| Artifact schema, `manifest.display`, API/download, frontend evidence, generated runs | `.codex/skills/quant-research-platform-maintenance/references/contracts-and-verification.md`; add `docs/architecture/engineering-standards.md` when module boundaries, validation, generated outputs, API/frontend behavior, or security boundaries are part of the task. |
| Diagnostics semantics, factor evaluation meaning, tradability, costs, liquidity, capacity, benchmarks, risk, lineage, or trading-adjacent wording | `docs/architecture/quant-research-standards.md` plus the contracts reference when artifacts/API/frontend are touched. |
| Code map edits | `.codex/skills/quant-research-platform-maintenance/references/code-map-flow.md` and the specific `docs/code-map/` files being changed. |
| README, architecture index, project-facing docs | The target doc, `AGENTS.md`, and only directly linked current docs needed to verify wording. |
| Decision task, architecture planning, or future capability boundary | `docs/architecture/target-architecture-and-capabilities.md`, `docs/architecture/next-stage-roadmap.md`, and the relevant standards. Ordinary coding tasks should not load target architecture by default. |
| Worktree, branch, or operations planning | `docs/architecture/next-stage-worktree-plan.md` while it remains active; if archived later, load it only as historical operations context. |
| Historical plans/specs under `docs/superpowers/` | Load only when tracing prior decisions, negative/rejection coverage, or explicitly cited historical context. Do not treat these files as current contract truth without confirmation from current docs, code, or tests. |
