# AGENTS.md

Rules for Codex work in `quant-research-platform`.

This file is for safety and execution boundaries only. Project narrative belongs in [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md), [REPORT.md](REPORT.md), [RESEARCH_CARD.md](RESEARCH_CARD.md), [PROJECT_STATUS.md](PROJECT_STATUS.md), and [docs/architecture/](docs/architecture/).

Background engineering principles are described in [docs/architecture/ai-native-engineering-principles.md](docs/architecture/ai-native-engineering-principles.md), but Codex execution boundaries are still governed by this `AGENTS.md`.

## 1. Scope

Project purpose:

- Local multi-market quantitative research platform.
- A-share is the first complete market module, backed by local private CSMAR data.
- Evaluation layer, artifact schemas, local API, downloads, and React evidence workbench live inside this project.
- Crypto and other markets are planned/disabled only until separately designed and tested; do not keep empty implementation packages for them.
- Do not add sibling workspace projects as runtime dependencies or active sources unless the user explicitly asks.
- Code and public-safe aggregate summaries may later be published after review.
- CSMAR raw/interim/processed data and private real reports remain private.

Current source maps and implementation entry points live in [docs/code-map/](docs/code-map/) and `.codex/skills/quant-research-platform-maintenance/references/contracts-and-verification.md`. Use [context-routing.md](.codex/skills/quant-research-platform-maintenance/references/context-routing.md) to choose task-specific context instead of broad-loading the repository.

- Do not create repo-level `apps/api` in the current Local Research Platform phase; it is a future deployable app shell.
- Do not create empty Future directories or empty market implementations merely as planning markers.

Hard boundaries:

- Do not publish, commit, copy, or expose CSMAR raw/interim/processed data.
- Do not log in to CSMAR, automate browser downloads, use CSMAR API, or handle credentials.
- Do not create, read, print, store, or commit credentials, tokens, cookies, secrets, or `.env` files.
- Do not connect vendor APIs, exchange APIs, broker APIs, web data APIs, paper trading, live trading, order routing, or production portfolio automation.
- Do not describe this project as deployed software, production automation, paper trading, live trading, investment advice, or a real performance guarantee.
- Do not claim factor validity, benchmark performance, universe coverage, or market alpha from preliminary diagnostics.
- Do not edit sibling projects or workspace-root files unless explicitly asked.

## 2. AI Development Guardrails

Codex may improve implementation efficiency, reduce repetitive work, and make local implementation improvements. Codex must not proactively change QRP system topology, core architecture, market scope, data scope, execution scope, or deployment scope.

Default to one research semantic layer per task: data, factor, signal, portfolio, evaluation semantics, trading-adjacent semantics, simulation semantics, or execution. Avoid cross-layer rewrites unless the user has approved that scope.

Public interfaces, artifact schemas, data schemas, factor output formats, evaluation semantics, trading-adjacent semantics, simulation semantics, module boundaries, new dependencies, new abstractions, and data storage structures require approval before implementation.

Small local edits such as typing improvements, logging cleanup, parser cleanup, lint/format fixes, focused test additions, and local duplicate-code reduction may proceed when they do not change contracts, schemas, public interfaces, or research semantics.

If an approved artifact/API/frontend/evaluation contract change has already happened, keep the linked surfaces synchronized: producer, writer, loader, API payloads/routes, download packaging, frontend consumers, tests, docs, and code maps must not be left half-updated.

### CURRENT_PHASE

Local Research Platform.

Current architecture style: Stable Core + High-Velocity Research Edge.

Current work must stay centered on factor research, evaluations, artifact generation, local evidence workflow, and A-share research reliability. Future capabilities may influence boundary design, but must not add complexity to the current implementation.

Out of scope for this phase: distributed runtime, workers/schedulers, billing/compliance, live trading, feature store, model registry, and runtime autonomous agents.

Contract rule: `manifest.display` is the only user-display contract. Do not bypass `manifest.display`, artifact flow, or evidence flow with frontend-only, backend-only, or task-local shortcuts.

### AI_STOP_CONDITIONS

Stop and request human confirmation when contract meaning is unclear, schema semantics are unclear, a task touches multiple architecture layers, a new abstraction/framework/dependency is proposed, work crosses the current phase boundary, public contracts change, or directory structure changes.

Goal / Overnight Run stop conditions include schema change, `manifest.display` change, public API or frontend shared contract change, directory restructuring, new dependency, new abstraction/framework, market/frequency/simulation/trading scope expansion, private/generated data edits, file deletion, archive or historical-doc physical moves, and unclear blast radius.

### FORBIDDEN_AUTONOMOUS_ACTIONS

Do not autonomously expand market scope, execution scope, deployment scope, runtime topology, public contracts, schemas, dependencies, directory layout, orchestration, plugin systems, or generalized governance infrastructure.

### TASK_PROTOCOL

Medium+ tasks must start from a task brief, implementation plan, scope declaration, stop conditions, and verification plan. The brief should identify task mode, context loaded, context routing, allowed scope, forbidden scope, relevant contracts, files to inspect, expected changes, required tests, definition of done, and risks.

Every task brief must declare execution settings: response language, recommended reasoning effort, branch plan, whether a worktree is needed with reason, whether subagents are allowed, planned subagent roles, whether challenge / objection is allowed, and whether a commit message recommendation is required.

Medium+ tasks must stop and wait for explicit human approval after producing the task brief, implementation plan, and scope declaration, and before implementation begins.

Decision Tasks must plan only and wait for approval before implementation.

Governed Batches may execute without per-step approval only after the user approves a clear bounded execution envelope.

Goal / Overnight Runs may execute autonomously only after the user approves a bounded execution envelope containing: goal, task mode, allowed scope, forbidden scope, files/directories in scope, contracts that must not change, permitted subagent roles, verification commands, stop conditions, and final report format.

Within an approved envelope, Codex may split work, use permitted subagents, read relevant files, edit in-scope files, run approved checks, and fix confirmed in-scope P0/P1/P2 issues without asking for each small step. Codex must stop and request confirmation when a stop condition is reached.

Small docs-only, typo, and single-file rules cleanup may run on a clean `main` when the user allows it. Code implementation, tests, diagnostics, pipeline, Factor Factory, API, artifact, contract-adjacent, schema-adjacent, and cross-layer work must use a semantic branch instead of implementing directly on `main`.

### Subagent Protocol

Subagents are allowed only inside an approved task envelope.

Each subagent must receive a narrow context packet containing: goal, assigned files/directories, allowed changes, forbidden changes, relevant contracts, relevant tests/checks, stop conditions, and expected output.

Subagents do not inherit project-level permissions from the parent goal, must not edit outside assigned scope, and must not independently modify shared contracts.

Shared contracts must be owned by the main agent or one explicit contract integrator.

## 3. Editable Area

Project-level Codex may edit:

```text
README.md
ARCHITECTURE.md
EVALUATION_INTERFACE.md
REPORT.md
RESEARCH_CARD.md
PROJECT_STATUS.md
AGENTS.md
data/README.md
docs/
.codex/skills/
src/
tests/
web/
config/
scripts/
reports/
pyproject.toml
.gitignore
```

Private data directories may be read or transformed only when the user has approved that task. Task-specific data approvals belong in the active implementation plan or PR summary, not as permanent AGENTS rules.

```text
data/raw_private/
data/interim_private/
data/processed_private/
reports/private_real/
```

Do not delete files or directories unless the user explicitly approves the exact path.

Product-side exception: the local workbench may expose `DELETE /api/runs/{source_alias}/{run_id}` only for user-created evaluation artifacts whose `summary.data_stage` is exactly `uploaded_panel`. The backend must resolve the run through configured run roots, require the run directory to be an immediate child of that root, require `manifest.json` to exist inside that directory, and reject all non-uploaded or out-of-root paths. This exception is limited to local uploaded run cleanup and does not authorize broad filesystem deletion, private data deletion, recursive cleanup scripts, or deletion outside configured run roots.

## 4. Paths and Portability

- Use project-relative paths in reusable documentation, code, scripts, and reports.
- Do not hardcode drive letters, home directories, user names, machine-specific absolute paths, interpreter paths, absolute symlinks, or shortcuts.
- Prefer forward slashes in documentation and config examples.
- In Python, build paths with `pathlib.Path` from the project root or current file location.
- Use `conda run -n analysis ...` in examples.
- If a command is OS-specific, label it clearly.

## 5. Environment and Commands

Approved default Conda environment:

```text
analysis
```

Rules:

- Do not use Conda `base` for project work.
- Use only Conda for Python dependency management.
- Do not use `pip` or `python -m pip`.
- Do not install, update, or remove dependencies unless the user has approved the exact boundary.
- Do not create additional environments without explicit user approval.

Common checks:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests
```

Frontend checks from `web/`:

```powershell
npm test -- --run --maxWorkers=1
npm run lint
npm run build
```

On Windows, `conda run` can emit OpenCL vendor noise even when the script succeeds. Use exit codes and test summaries, not the noise line alone, to judge success.

## 6. Data Responsibilities

CSMAR data layers:

```text
data/raw_private/csmar/          original CSMAR exports
data/interim_private/csmar/      normalized audit layer
data/processed_private/csmar/    research-ready Parquet
reports/private_real/            private quality, pipeline, and evaluation reports
```

Current processed contract:

- `panel_full/research_panel_<start>_<end>.parquet` is the full-sample research panel.
- `price_volume_factors_full/price_volume_factors_<start>_<end>.parquet` is the slim factor and label table.
- Slim factor files must keep `stock_code`, `trade_date`, tradability columns, common liquidity/market-cap columns, factor columns, and forward label columns.

Platform evaluation contract:

- Market modules produce explicit DataFrames or Parquet inputs.
- Evaluation layer receives `EvaluationSpec` and runs `run_evaluation(spec)`.
- Evaluation diagnostics belong under `evaluations/diagnostics/` as post-run explanation logic, not as a top-level platform `diagnostics/` domain.
- A-share full/private evaluation runs use the current A-share private evaluation entry point documented in the project maintenance reference; generic `evaluate-parquet` is only for external/local Parquet adapter checks or non-A-share inputs.
- API-visible run identity is `(source_alias, run_id)`. Preserve `source_alias` across API, downloads, cache, comparison, and frontend routing, while keeping portable artifacts free of local API aliases unless the current schema explicitly allows them.
- `manifest.display` is the source of truth for API evidence, CSV preview, direct downloads, ZIP downloads, frontend tables, frontend charts, and download presets.
- Factor Factory has its own artifact contract; do not conflate it with evaluation-run artifact contracts.
- Do not create frontend-only or backend-only artifact shortcuts that bypass `manifest.display`.
- When artifact, API, or frontend contracts change, update the linked producer/API/frontend/test/doc surfaces in the same round. Use the project maintenance skill for detailed review and verification sequencing. Do not weaken validation to keep stale generated runs visible; regenerate stale runs instead.

Frontend workbench structure:

- `web/src/shell/` owns top-level React workbench composition, view mode, and cross-feature coordination.
- `web/src/features/*/` owns feature-local UI, hooks, state helpers, and tests.
- `web/src/shared/*` owns reusable API types/helpers, shared feature state types, UI primitives, formatting, downloads, and test fixtures.
- Allowed dependency direction is `shell -> features -> shared`.
- `web/src/shared/*` must not import from `features/*` or `shell/*`.
- Cross-feature imports should stay rare; prefer shell composition or moving genuinely shared helpers into `shared/`.
- Do not add broad barrel exports unless a feature has a stable public surface and the exports are intentionally curated.

## 7. Testing

Run the narrowest relevant tests first.

Contract rules that can be mechanically checked should live in pytest or Vitest rather than only in agent instructions. Use the project maintenance skill for detailed review surfaces and verification sequencing.

Artifact/API/frontend/A-share/Factor Factory contract work must use the project maintenance skill and cross-check the referenced engineering and quant research standards before changing public contracts, diagnostics semantics, generated outputs, or trading-adjacent wording.

For frontend Playwright acceptance of API-backed workbench flows, follow the project maintenance skill: start both local services, record ownership, stop only recorded processes, and do not count an API-offline `Failed to fetch` state as acceptance evidence unless the task is explicitly UI-only.

Do not run slow, network-heavy, destructive, or environment-changing commands without explicit approval.

## 8. Documentation and Context Routing

Use [README.md](README.md) as the project entry, [PROJECT_STATUS.md](PROJECT_STATUS.md) for current status and root handoff, [docs/code-map/](docs/code-map/) for implementation maps, and [context-routing.md](.codex/skills/quant-research-platform-maintenance/references/context-routing.md) to choose task-specific docs. Use `.codex/skills/quant-research-platform-maintenance/SKILL.md` for artifact/API/frontend/A-share/Factor Factory maintenance workflow.

If GPT Pro or other external tools are used, they must receive an explicit context packet or public link. Context packets are managed by the root workflow; do not create a project-local packet cache unless approved.

Root handoff rule: if this project changes its current topic, phase, milestone, blocker, next step, public readiness, or data/API/Git/Conda/privacy boundary, record an `open` signal in [PROJECT_STATUS.md](PROJECT_STATUS.md). Do not edit workspace-root planning or status files from this project unless explicitly asked.

Use the workspace learning-day convention for project status, task summaries, research evidence, and planning handoffs:

- Late-night work after midnight but before the user sleeps still belongs to the previous calendar day.
- Morning updates after 08:00 belong to the current calendar day.
- If date ownership is ambiguous, ask before creating or updating dated project records.

## 9. Forbidden Operations

Do not:

- Delete files or directories without exact explicit approval.
- Use recursive or wildcard deletion commands without explicit approval.
- Use unapproved external data.
- Connect vendor APIs, exchange APIs, broker APIs, web data APIs, paper trading, live trading, order routing, or production portfolio automation.
- Create, read, print, or store credentials.
- Install, update, or remove dependencies outside approved boundaries.
- Use `pip` in this project.
- Hardcode absolute paths in reusable files.
- Copy private notes, copyrighted materials, full problem banks, or copied solutions.
- Deploy the website or expose the local API outside the machine without a separate design and approval.
