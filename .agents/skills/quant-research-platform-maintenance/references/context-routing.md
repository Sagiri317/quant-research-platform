# Context Routing

Use this reference to choose the smallest useful context for QRP Clean Rewrite
work. It is routing guidance, not a separate approval layer.

## Rules

- Start with targeted `rg`, then read narrow file ranges.
- Do not broad-load generated outputs, private data, old branch history, or full
  docs unless the approved envelope requires it.
- Subagents read only assigned files/directories and report findings; they do
  not edit, install, commit, push, or merge.
- If context expansion would touch private data, dependencies, deployment,
  trading, or unclear contract meaning, stop and report.

## Task Routing

| Task | Load |
|---|---|
| Current context primer | `.agents/briefs/current-qrp-status.md` |
| Any medium+ QRP task | `AGENTS.md`, user envelope, and the matching row below |
| Contracts/specs/catalogs/storage | `references/contracts-and-verification.md`, `src/quant_research_platform/contracts/`, `storage/`, `catalog/` |
| Data product/interim migration | `catalog/data_products.py`, `sources/`, `docs/data/README.md`, relevant tests |
| A-share market semantics | `markets/implementations/a_share/`, `docs/code-map/a-share-data-and-pipeline.md`, relevant tests |
| Research datasets, evaluation, prediction | `research`, `reports`, `docs/code-map/reports-api.md`, relevant tests |
| API/CLI | `interfaces/api/`, `interfaces/cli/`, `docs/api/README.md`, `tests/interfaces/` |
| Workflows | `workflows/`, `contracts/workflows.py`, `tests/workflows/` |
| Agents/proposals | `agents/`, `contracts/`, relevant tests |
| Code maps | `references/code-map-flow.md` and changed `docs/code-map/` files |
| Commit/push | `git status`, `git diff`, verification outputs, and changed-file list |

For long goal work, follow the approved envelope and the skill's Goal Workflow.
