---
name: quant-research-platform-maintenance
description: Use for QRP Clean Rewrite maintenance: 0.1 contracts, source topology, storage/catalog boundaries, A-share data flow, research datasets, reports, API/CLI, private-boundary checks, docs/status sync, branch/commit/push workflow, and read-only subagent coordination.
---

# Quant Research Platform Maintenance

## Purpose

Keep QRP Clean Rewrite `0.1.0` aligned across code, tests, docs, and `.agents`
governance. Current public API/spec/contract version is `0.1`.

This skill is a routing and safety layer. It does not approve private-data
exposure, dependency installs, deployment, website UI work, scheduler/worker
runtime, broker/exchange connectors, paper/live trading, or external LLM APIs.

## Start Here

1. Read project `AGENTS.md`.
2. Read `.agents/briefs/current-qrp-status.md` for the current compact context.
3. Use `references/context-routing.md` to choose the smallest relevant context.
4. Use `references/contracts-and-verification.md` when a task touches 0.1
   contracts, storage/catalog boundaries, source/market semantics, reports,
   API/CLI, or private-boundary checks.

## Execution Settings

- Default response language: Chinese; keep code, paths, commands, field names,
  API paths, and error strings in English.
- Reasoning: use high reasoning for cross-layer implementation and highest
  available reasoning for clean rewrite architecture or long goal work.
- Branch plan: use `feature/<task-name>` for code, tests, architecture, API/CLI,
  data-flow, reports, or governance work.
- Worktree: use a project-local Git worktree when the main checkout must remain
  clean or a long goal needs isolation.
- Subagents: read-only unless the user explicitly approves otherwise.

## Goal Workflow

Use this only inside an approved bounded envelope.

1. Confirm branch/worktree and clean status.
2. Keep shared contracts under the main agent.
3. Split read-only subagents by disjoint review scopes only.
4. Implement in small slices.
5. Run narrow tests after each slice.
6. Before commit/push, run targeted pytest, `ruff`, `git diff --check`, and the
   clean rewrite stale-term gate.
7. Stop on unexplained regression, private-boundary ambiguity, new dependency,
   deployment/trading scope, unapproved private-data action, or unclear blast
   radius.

## Private Output Checklist

Apply this whenever a command, API handler, workflow, report writer, exporter,
or metadata writer can persist row-level or private-derived output.

- Private-derived row-level output must write only under approved private roots.
- Prediction/evaluation/diagnostics/portfolio run outputs write under
  `reports/private_real/<market>/<run_type>/<run_id>/`.
- Metadata, stdout, stderr, API errors, reports, manifests, and summaries must
  be sanitized.
- Do not expose local absolute paths, raw/interim/processed row samples,
  stock-code samples, vendor/source markers, credentials, tokens, or raw
  tracebacks with local/private context.

## Current Clean Topology

```text
contracts/
storage/
catalog/
sources/
markets/
research/
workflows/
compute/
agents/
interfaces/
reports/
runtime/
```

The first active market is `markets/implementations/a_share`. Public-facing
objects use `data_id`, `research_dataset_id`, `workflow_id`, or `artifact_id`
according to their layer semantics. Future markets and sources must not be
added as empty packages.
