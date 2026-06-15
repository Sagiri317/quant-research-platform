# AGENTS.md

Execution rules for AI agents working in `quant-research-platform`.

This file is intentionally short and stable. It defines hard boundaries, task routing, and stop conditions. Load detailed workflow only from `.agents/` when the task needs it.

## 1. Project boundary

QRP is a local quantitative research platform. It is not investment advice, a paper-trading system, a live-trading system, a broker/exchange connector, or production deployment infrastructure.

Current development baseline: `feature/qrp-clean-rewrite`. Do not treat `main` as the active implementation target unless the user explicitly asks for a merge/release task.

Current source topology:

```text
src/quant_research_platform/
  contracts/ storage/ catalog/ sources/ markets/ research/
  workflows/ compute/ agents/ interfaces/ reports/ runtime/
```

Current public surface:

```text
GET /api/0.1/catalog/factors
GET /api/0.1/catalog/labels
GET /api/0.1/research-datasets/{dataset_id}/requirements
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

Current private local CLI is narrow: `qrp-private schema-inventory --project-root . --output <path>`. It is not a public API expansion.

## 2. Non-negotiable principles

- Semantic closure: keep contracts, names, ownership, tests, docs, and artifacts aligned.
- Naming is architecture: names must express owner layer and lifecycle, not transient implementation detail.
- Replace over parallel: remove replaced paths; do not add compatibility aliases, wrappers, re-exports, or legacy shims by default.
- Minimal public API: new public API, CLI, registry, store, facade, artifact type, or execution mode is forbidden unless explicitly approved.
- Evidence over claims: never report `PASS` unless the command or check actually ran and produced that result. Preserve `FAIL` and `NOT_RUN` honestly.
- Private-data safety: no row-level private data, vendor markers, stock-code samples, local absolute paths, credentials, tokens, or raw tracebacks with local/private context.
- Layer-local change: new work should mostly change the owner layer. Cross-storage/evaluation/report/workflow/interface edits require architecture closure first.
- No handoff-driven architecture: `.run` evidence may inform work, but handoff artifacts do not define architecture.
- Extensible without abstraction creep: future ML/DL, diagnostics, factor factory, multi-market, high-frequency, and frontend work must not create premature generic frameworks.

## 3. Agent governance paths

- Project governance root: `.agents/`. Keep it agent-neutral for Codex, GPT, Claude, and other agents.
- global `.codex/`: user-level Codex state is read-only external evidence when explicitly supplied. Do not modify it and do not copy raw content into the repo.
- Repo-local `.codex/`: do not create it by default. Codex-specific hooks/config belong only in explicit optional/draft adapter proposals.
- Do not add project `CLAUDE.md`, `.claude/`, hooks, MCP servers, plugins, or subagent configs by default. Use draft-only docs unless the user explicitly approves implementation.

## 4. Task router gate

Before editing, classify the task:

```text
small fix | standard implementation | goal closure | independent review |
handoff packaging | merge gate | release gate | docs/status sync |
governance redesign | apply-only validation | ambiguous
```

Then load the smallest matching context:

- Always read this file and the user envelope.
- For QRP task routing, use `.agents/skills/quant-research-platform-maintenance/SKILL.md`.
- For current facts, use `.agents/briefs/current-qrp-status.md` and `.agents/briefs/user-principles.md`.
- For task modes and gates, use `.agents/skills/quant-research-platform-maintenance/references/task-modes.md` and `governance-gates.md`.
- For implementation ownership, use `context-routing.md`, `owner-layer-routing.md`, code, and tests.

If the task is ambiguous, stop and ask. Do not silently convert review, packaging, merge, or governance work into business implementation.

## 5. Stop conditions

Stop before editing or before continuing if any item appears without explicit approval:

- Dirty or unclear working tree, branch, base branch, or user change ownership.
- Public API/CLI/schema/report display contract change.
- New public registry, store, facade, service, compatibility layer, execution mode, artifact type, or dependency.
- New top-level package, directory relocation, or cross-layer redesign.
- Private data access not explicitly approved by the task envelope.
- Deployment, scheduler, worker, model registry, feature store, external LLM API, vendor API, exchange/broker connector, paper/live trading, or trading-adjacent automation.
- Failed/unknown validation that would need to be represented as `PASS` to proceed.
- Repeated unclear test failure or performance regression.

## 6. Verification and truthfulness

Use the narrowest relevant checks first, then broaden only when the touched surface requires it. For medium+ changes, report:

```text
changed files | checks actually run | checks not run | PASS/FAIL/NOT_RUN |
private-safety result | public-surface result | docs/status sync result |
remaining risks
```

Before commit or handoff, run `git diff --check` and the relevant deterministic governance/script gates. If a check was skipped, say why; do not mark it as passing.

## 7. Data and artifact safety

Private roots are local-only:

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
data/research/a_share/
reports/private_real/
```

Review bundles, handoff packages, and public-safe artifacts must not include parquet, csv, pkl, pickle, feather, arrow, raw vendor exports, row-level private data, credentials, tokens, or old handoff zips/json unless a specific package rule explicitly allows a sanitized summary.

## 8. Git safety

Run `git status --short --branch` before editing. Do not discard user changes. Do not amend, rebase, squash, force-push, delete branches, merge to `main`, or push `main` unless the exact operation is explicitly approved. Default branch for implementation or governance work is `feature/<task-name>`.
