# Current QRP Status

This is a short current task context primer for ordinary QRP Codex tasks. It is
not a Git log, latest HEAD tracker, changelog, or replacement for
`PROJECT_STATUS.md`.

Read this brief before broad-loading `PROJECT_STATUS.md`, `README.md`, or the
roadmap when a task only needs current project orientation. Re-check
`PROJECT_STATUS.md` before changing current topic, blocker, milestone, next
step, public readiness, data/API/Git/Conda/privacy boundary, or handoff state.

Ordinary commits, docs wording edits, pushes, and merges do not require an
update here unless they change future Codex task context, safety boundaries, or
the active next-step decision.

## Current Phase

- Local Research Platform.
- Current architecture style: Stable Core + High-Velocity Research Edge.
- Current work stays centered on factor research, evaluations, artifact generation, local evidence workflow, and A-share research reliability.

## Status-Relevant Milestones

- N5 private real-data smoke passed: `n5_private_smoke_2025h2_rev120_v1`.
- Pipeline failed-run reliability MVS and failed-run disk-recovery sanitization hardening have merged to `main`; API duplicate `run_id` / rerun guard is implemented in current code.
- N6 prediction handoff guardrails, synthetic handoff smoke, local numpy/pandas ML baseline trainer, N6 prediction handoff MVS, private marker helper centralization, and `ml baseline train` CLI pre-read private-output validation have merged to `main`.
- Agent workflow layout has moved to `.agents`: the maintenance skill lives under `.agents/skills/quant-research-platform-maintenance/`, task templates live under `.agents/skills/quant-research-platform-maintenance/references/templates/`, and this brief lives under `.agents/briefs/current-qrp-status.md`.
- These facts are status pointers, not full evidence records.
- Re-check `PROJECT_STATUS.md` before changing current topic, blocker, milestone, next step, or handoff state.

## Milestone Context

- N5 v1 private smoke demonstrates the current descriptive proxy diagnostics path on private A-share data.
- Pipeline failed-run reliability now includes the original failed-run MVS, disk-loaded `runtime_summary.json` / `checkpoints.json` shared private-marker sanitization, and API-created duplicate `run_id` rejection before writing new pipeline outputs.
- N6 now has initial handoff guardrails, a synthetic smoke path, a local baseline trainer, OOS / embargo / fold metadata, `ml baseline train` CLI handoff, private-output boundary guard, shared private-marker helper, and CLI pre-read private-output validation. This is not a complete ML pipeline, deep learning backend, model registry, feature store, A-share prediction adapter, paper/live trading system, deployment, scheduler, worker, or external connector.
- Task-mode template details live in `.agents/skills/quant-research-platform-maintenance/references/templates/`, not in this brief; one-line or underspecified QRP prompts route through `auto-task-router.md`.

## Current Next Step

- Decide the next N6 slice: multi-fold / walk-forward extension, A-share prediction adapter design, or continued local synthetic handoff.
- If continuing pipeline reliability, pick one new small MVS such as retry, resume, destructive overwrite, or global API error envelope; do not reopen the completed failed-run sanitization hardening or duplicate `run_id` guard.
- Keep this as an audit/decision point; do not jump directly into broad implementation.

## Known Limitations

- Retry, resume, destructive overwrite, and global API error envelope remain unimplemented.
- N5 v1 is descriptive proxy diagnostics only.
- N5 v1 is not full risk modeling, capacity modeling, simulation, trading, validated alpha, or investment advice.
- N6 baseline work is not a complete ML platform, alpha proof, real risk/capacity model, trading simulation, paper/live trading evidence, deployment, model registry, feature store, or external connector.
- Generated/private evidence remains local and should not be copied into public docs.
- This brief is a short context primer, not a replacement for active project docs.

## Hard Reminders

- `manifest.display` is the user-display contract for evidence, previews, downloads, charts, and presets.
- API-visible run identity is `(source_alias, run_id)`.
- Keep private CSMAR data and private real reports out of commits, logs, portable artifacts, and public summaries.
- Do not connect vendor, broker, exchange, paper-trading, live-trading, or external market-data APIs.
- Do not use Conda `base` for project work; the approved project environment is `analysis`.
- Do not delete files or directories without exact explicit approval.
- Ordinary tasks should use narrow context first; goal or overnight work needs an approved envelope.

## Active Docs Pointers

- `AGENTS.md`: execution boundaries, stop conditions, editable area, data/privacy rules.
- `README.md`: project entry and Codex quick start.
- `PROJECT_STATUS.md`: current status, handoff, blockers, next steps.
- `.agents/skills/quant-research-platform-maintenance/SKILL.md`: maintenance workflow.
- `.agents/skills/quant-research-platform-maintenance/references/context-routing.md`: task-specific context loading.
- `.agents/skills/quant-research-platform-maintenance/references/templates/`: task-mode templates.
