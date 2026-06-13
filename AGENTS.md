# AGENTS.md

Rules for Codex work in `quant-research-platform`.

This file is for execution boundaries only. Project facts belong in
`README.md`, `PROJECT_STATUS.md`, `docs/architecture/`, `docs/code-map/`,
`docs/data/`, `docs/api/`, and `.agents/`.

## 1. Scope

Project purpose:

- Local quantitative research platform.
- A-share is the first enabled market module, backed by local private vendor
  exports.
- Current implementation phase is QRP Clean Rewrite `0.1.0`.
- Public API/spec/contract version is `0.1`.
- Public user-facing data objects are factors, labels, and research dataset
  requirements.
- Code and public-safe aggregate summaries may later be published after review.
- Raw/interim/processed private data and private real reports remain private.

Hard boundaries:

- Do not publish, commit, copy, or expose raw/interim/processed private data.
- Do not log in to vendors, automate browser downloads, use vendor APIs, or
  handle credentials.
- Do not create, read, print, store, or commit credentials, tokens, cookies,
  secrets, or `.env` files.
- Do not connect vendor APIs, exchange APIs, broker APIs, paper/live trading,
  order routing, production portfolio automation, deployment, scheduler, worker,
  model registry, feature store, or external LLM APIs.
- Do not describe this project as deployed software, investment advice, paper
  trading, live trading, or a performance guarantee.
- Do not create empty future market/source packages as planning markers.
- Do not edit sibling projects or workspace-root files unless explicitly asked.

## 2. Clean Rewrite Topology

Target source topology:

```text
src/quant_research_platform/
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

The current clean rewrite does not keep retired top-level source packages as
active targets. Use the owners above instead.

## 3. Data And Reports

Private data/report roots:

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
data/research/a_share/
reports/private_real/
```

Processed canonical key:

```text
market_id
asset_id
timestamp
```

`stock_code` and `trade_date` may remain source or migration fields, but they are
not processed canonical keys.

Prediction, evaluation, diagnostics, and portfolio run outputs belong under
`reports/private_real/<market>/<run_type>/<run_id>/`, not under processed data.

## 4. API And CLI

The clean API uses `/api/0.1/...` and no version-letter prefix. The current route
set is:

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

The clean CLI surface is:

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

These public commands are read-only. Source/interim/processed materialization is
internal, and private real-data execution is not exposed through API/CLI in
`0.1.0`.

Retired command and endpoint names are not compatibility requirements.

## 5. Agent Governance

`AGENTS.md` and `.agents/` are governance files and must be retained. They may be
updated to match the clean rewrite, but do not delete them or change governance
rules without explicit user approval.

For QRP tasks, use:

- `.agents/briefs/current-qrp-status.md`
- `.agents/skills/quant-research-platform-maintenance/SKILL.md`
- `.agents/skills/quant-research-platform-maintenance/references/context-routing.md`
- `.agents/skills/quant-research-platform-maintenance/references/contracts-and-verification.md`

## 6. Task Protocol

Medium+ tasks must start from a scope declaration, implementation plan,
verification plan, and stop conditions unless the user has already provided and
approved a bounded execution envelope.

Within an approved goal envelope, Codex may split work, use subagents, edit
in-scope files, run approved checks, fix in-scope defects, commit, and push the
feature branch. Stop when a listed stop condition is reached.

The main agent owns integration, verification, commits, and pushes.

## 7. File And Deletion Safety

Before editing in this Git repository, run `git status` or an equivalent status
check.

Do not discard user changes. Do not run destructive Git history operations,
force pushes, broad cleanup commands, or dependency installs without explicit
approval.

Private data directories may be read or transformed only when the task envelope
approves that data action. Do not print row samples, stock-code samples, private
absolute paths, vendor markers, secrets, tokens, or raw tracebacks with
local/private context.

## 8. Environment

Use the Conda environment defined in `environment.yml`. Keep setup and common
check commands in `README.md`.

## 9. Testing And Verification

Run the narrowest relevant tests first, then broaden when the touched surface is
internally consistent.

Contract rules that can be mechanically checked should live in tests or scripts.
Before commit/push, run targeted pytest, `ruff`, `git diff --check`, and a stale
term/path gate relevant to the clean rewrite.

## 10. Documentation

Use `README.md` as the project entry, `PROJECT_STATUS.md` for current status and
handoff, `docs/code-map/` for implementation maps, `docs/architecture/` for
current architecture boundaries, `docs/data/` for public-safe data layout notes,
and `docs/api/` for the 0.1 API/CLI surface.

Do not put commit SHAs, one-time run evidence, private paths, or debug logs in
durable docs.
