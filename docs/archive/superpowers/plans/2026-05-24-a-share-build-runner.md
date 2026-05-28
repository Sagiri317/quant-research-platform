# Superseded Plan: A-share Build Runner

This plan has been superseded by the `start_from` interface refactor decided on 2026-05-24.

Current contract:

- `POST /api/pipeline-snapshots` writes configuration snapshots only.
- `POST /api/pipeline-runs` executes the A-share pipeline.
- Executable runs use `start_from`:
  - `existing_factors`
  - `rebuild_factors`
  - `rebuild_panel`
  - `rebuild_interim`
- Every executable run continues through evaluation.
- The canonical runner is `src/quant_research_platform/pipelines/a_share/runner.py::run_a_share_pipeline`.

Do not use this file as the implementation plan for new work. Use the current code, tests, `PROJECT_STATUS.md`, and `docs/architecture/next-stage-roadmap.md` instead.
