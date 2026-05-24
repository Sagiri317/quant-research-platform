# Next-Stage Worktree Plan

This document is historical. The active N1 interface decision is now recorded in [next-stage-roadmap.md](next-stage-roadmap.md), [PROJECT_STATUS.md](../../PROJECT_STATUS.md), and the pipeline tests.

Current N1 contract:

- Snapshot-only endpoint: `POST /api/pipeline-snapshots`.
- Executable endpoint: `POST /api/pipeline-runs`.
- Executable payload uses `start_from`.
- Supported start layers: `existing_factors`, `rebuild_factors`, `rebuild_panel`, `rebuild_interim`.
- Every executable run continues through evaluation.
- Canonical runner: `src/quant_research_platform/pipelines/a_share/runner.py::run_a_share_pipeline`.

Do not treat this file as the current implementation checklist.

N3 Artifact ZIP packaging has already merged to `main`. N4 frontend feature split and N2 eligibility / tradability diagnostics remain separate follow-up workstreams. New work should use [next-stage-roadmap.md](next-stage-roadmap.md) as the planning source and keep external pipeline language to snapshot vs run plus `run.start_from`.
