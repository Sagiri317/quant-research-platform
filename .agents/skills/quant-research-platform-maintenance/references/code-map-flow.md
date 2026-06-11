# Code Map Flow Maintenance

Load this reference when changing `docs/code-map/platform-flow.md`.

## Required Terms

- QRP Clean Rewrite
- `0.1.0`
- `/api/0.1`
- `ResearchDatasetSpec`
- `WorkflowSpec`
- `reports/private_real`

## Required Relationships

- Public catalog objects and workflow specs reference concrete artifacts by
  `data_id`, research inputs by `research_dataset_id`, and workflows by
  `workflow_id`.
- A-share processed layers live under `markets/implementations/a_share/`.
- Research datasets, evaluations, models, and predictions live under `research/`.
- Run evidence writes under `reports/private_real`.
- `interfaces/api/app.py` exposes only the 0.1 route set.
- `interfaces/cli/commands.py` exposes only the 0.1 CLI groups.

## Checks

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.ps1
git diff --check
```
