# Docs / Status Sync Template

Use when current truth changes and active docs need synchronization.

## Active docs map

- `README.md`: project entry and stable public boundary.
- `PROJECT_STATUS.md`: current operational truth and current blockers/next steps.
- `docs/api/README.md`: public API/CLI contract.
- `docs/code-map/`: implementation maps.
- `docs/architecture/`: durable architecture and governance rationale.
- `.agents/briefs/`: short operational context and user principles.
- `.agents/skills/`: execution workflow and deterministic helper scripts.

## Rules

- Do not write one-time run logs, real commit SHAs, private paths, or debug traces into durable docs.
- Do not preserve stale N1/N2/N3 branch claims as current truth.
- Do not update all docs by default; update only docs whose current truth changed.
- Historical or superseded notes should be marked as such or moved only when the task approves archival work.

## Checks

```text
git diff --check
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only
```
