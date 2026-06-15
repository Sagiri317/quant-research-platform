---
name: quant-research-platform-maintenance
description: Use for QRP governance, maintenance, implementation routing, review, handoff packaging, merge gates, private-safety checks, public API/CLI contract checks, owner-layer routing, and Codex/GPT/Claude agent workflow discipline. Do not use for live trading, deployment, vendor login, broad private-data access, or external API integration.
---

# Quant Research Platform Maintenance

This skill is the operational routing layer for QRP. It keeps code, tests, docs, `.agents/`, `.run` evidence, and final reports aligned without loading every rule into every task.

## Load order

1. Read repo `AGENTS.md`.
2. Read the user task envelope.
3. For current facts, read `.agents/briefs/current-qrp-status.md`.
4. For user principles, read `.agents/briefs/user-principles.md` when the task touches architecture, naming, governance, public surface, or cross-layer scope.
5. Use `references/task-modes.md` to classify the task.
6. Use `references/context-routing.md` to load only the smallest relevant context.
7. Use `references/governance-gates.md` for mandatory gates.
8. Use `references/contracts-and-verification.md` for public API/CLI, artifacts, data, report, private-safety, or verification semantics.
9. Use `references/owner-layer-routing.md` before changing or routing P2/P3 items.

## Default execution settings

- Language: final user-facing responses are Chinese by default; keep code, paths, commands, API routes, field names, and error strings in English.
- Branch: use `feature/<task-name>` for implementation, architecture, docs/rules, or governance changes unless the user explicitly approves another branch.
- Context: targeted `rg` first, then narrow file reads. Do not broad-load private data, generated outputs, old handoff packages, or all docs by default.
- Subagents: review-only unless the user explicitly approves write-capable subagents. The main agent owns shared contracts, integration, verification, commits, pushes, and handoff truthfulness.
- Hooks/MCP/plugins/repo-local `.codex`: draft-only unless explicitly approved.

## Task modes

Classify before editing:

```text
small-fix | standard-implementation | goal-closure | independent-review |
handoff-packaging | merge-gate | release-gate | docs-status-sync |
governance-redesign | apply-only-validation | ambiguous
```

If classification is ambiguous, stop and ask. Do not convert a review, handoff, merge, packaging, or governance task into business implementation.

## Required gates

For every medium+ task, run the relevant gates from `references/governance-gates.md`:

```text
Task Router | Scope | Context | Owner Layer | Blast Radius | Red Test |
Same Issue Family Scan | Artifact Truth | Public Surface | Private Safety |
Memory/Performance Awareness | Docs/Status Sync | .run Tooling |
Script Portability | Handoff | Final Truthfulness
```

## Truth rules

- `PASS` means the check ran and passed.
- `FAIL` means the check ran and failed; do not relabel it.
- `NOT_RUN` means no run evidence exists.
- If a command was not available, mark the command as `NOT_RUN` or `BLOCKED`, not `PASS`.
- Final reports must list checks actually run, checks not run, and residual risks.

## Private output rule

Any command, API handler, workflow, report writer, exporter, package builder, metadata writer, or handoff tool that can persist private-derived row-level output must use the Private Safety Gate. Public-safe bundles must exclude private data and blocked file types.

## Validation scripts

Use the cross-platform scripts in `scripts/` when relevant:

```text
python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --help
python .agents/skills/quant-research-platform-maintenance/scripts/package_safety_check.py --help
python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --help
```

Scripts are deterministic helpers. They do not replace tests, code review, or task-specific validation.
