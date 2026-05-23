# AGENTS.md

Rules for Codex work in `quant-research-platform`.

This file is for safety and execution boundaries only. Project narrative belongs in [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md), [REPORT.md](REPORT.md), [RESEARCH_CARD.md](RESEARCH_CARD.md), [PROJECT_STATUS.md](PROJECT_STATUS.md), and [docs/architecture/](docs/architecture/).

## 1. Scope

Project purpose:

- Local multi-market quantitative research platform.
- A-share is the first complete market module, backed by local private CSMAR data.
- Evaluation layer, artifact schema `0.5`, local API, downloads, and React evidence workbench live inside this project.
- Crypto and other markets are planned/disabled only until separately designed and tested; do not keep empty implementation packages for them.
- Sibling project `../alpha-research-core` is historical reference only. Do not make it a runtime dependency or active source after migration unless the user explicitly asks.
- Code and public-safe aggregate summaries may later be published after review.
- CSMAR raw/interim/processed data and private real reports remain private.

Current structure:

- `src/quant_research_platform/evaluations/`: evaluation contracts, panel validation, run orchestration, and result objects.
- `src/quant_research_platform/kernels/`: DataFrame and tensor helper implementations.
- `src/quant_research_platform/artifacts/`: schema `0.5` writer, loader, inventory, comparison, and `manifest.display`.
- `src/quant_research_platform/interfaces/api/`: local FastAPI evidence API routes, service layer, CSV preview, and downloads.
- `src/quant_research_platform/interfaces/cli/`: `qrp` / `python -m quant_research_platform.interfaces.cli` commands.
- `src/quant_research_platform/datasets/`: generic CSV / Parquet readers and dataset access helpers.
- `src/quant_research_platform/markets/implementations/a_share/`: CSMAR-backed A-share market implementation, including data fields, factors, labels, validation, runtime helpers, and evaluation adapter.
- `src/quant_research_platform/pipelines/a_share/`: A-share workflow orchestration, runner, checkpoint, snapshots, and step registry.
- `web/`: React/Vite local evidence workbench.
- `scripts/`: local-only PowerShell launch scripts, build checks, and acceptance helpers.
- `docs/architecture/`: architecture blueprint, future capabilities, standards, Phase 1 refactor design, implementation checklist, and directory-rename handoff.
- `docs/code-map/`: source reading map and contract handoff notes.

- Do not create repo-level `apps/api` in Phase 1; it is a future deployable app shell for Phase 2.
- Do not create empty Future directories or empty market implementations merely as planning markers.

Hard boundaries:

- Do not publish, commit, copy, or expose CSMAR raw/interim/processed data.
- Do not log in to CSMAR, automate browser downloads, use CSMAR API, or handle credentials.
- Do not create, read, print, store, or commit credentials, tokens, cookies, secrets, or `.env` files.
- Do not connect vendor APIs, exchange APIs, broker APIs, web data APIs, paper trading, live trading, order routing, or production portfolio automation.
- Do not describe this project as deployed software, production automation, paper trading, live trading, investment advice, or a real performance guarantee.
- Do not claim factor validity, benchmark performance, universe coverage, or market alpha from preliminary diagnostics.
- Do not edit sibling projects or workspace-root files unless explicitly asked.

## 2. Editable Area

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
notebooks/
reports/
pyproject.toml
.gitignore
```

Private data directories may be read or transformed only when the user has approved that task:

```text
data/raw_private/
data/interim_private/
data/processed_private/
reports/private_real/
```

Do not delete files or directories unless the user explicitly approves the exact path.

Product-side exception: the local workbench may expose `DELETE /api/runs/{source_alias}/{run_id}` only for user-created evaluation artifacts whose `summary.data_stage` is exactly `uploaded_panel`. The backend must resolve the run through configured run roots, require the run directory to be an immediate child of that root, require `manifest.json` to exist inside that directory, and reject all non-uploaded or out-of-root paths. This exception is limited to local uploaded run cleanup and does not authorize broad filesystem deletion, private data deletion, recursive cleanup scripts, or deletion outside configured run roots.

## 3. Paths and Portability

- Use project-relative paths in reusable documentation, code, scripts, notebooks, and reports.
- Do not hardcode drive letters, home directories, user names, machine-specific absolute paths, interpreter paths, absolute symlinks, or shortcuts.
- Prefer forward slashes in documentation and config examples.
- In Python, build paths with `pathlib.Path` from the project root or current file location.
- Use `conda run -n analysis ...` in examples.
- If a command is OS-specific, label it clearly.

## 4. Environment and Commands

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

## 5. Data Responsibilities

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
- Evaluation artifacts use schema `0.5`.
- A-share full/private evaluation runs should use `python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation`, not the generic `evaluate-parquet` CLI. Use `--start-year/--end-year` to locate the processed factor file and `--start-date/--end-date` for `trade_date` filtering.
- Use generic `evaluate-parquet` only for external/local Parquet adapter checks or non-A-share inputs. Do not treat a temporary sliced Parquet plus `evaluate-parquet` as proof that the A-share adapter pipeline works.
- The current frontend Pipeline Builder defaults to a configuration snapshot flow through `POST /api/pipeline-runs`; the backend also supports `execution_mode=processed_existing` as a lightweight runner over existing processed A-share factor data.
- API-visible run identity is `(source_alias, run_id)`. Preserve `source_alias` in run lists, evidence routes, CSV preview, direct downloads, ZIP cache paths, comparison payloads, and frontend routing.
- `manifest.display` is the source of truth for API evidence, CSV preview, direct downloads, ZIP downloads, frontend tables, frontend charts, and download presets.
- Do not create frontend-only or backend-only artifact shortcuts that bypass `manifest.display`.

## 6. Testing

Run the narrowest relevant tests first.

Important test surfaces:

- CSMAR field contracts, filename parsing, metadata, and field mapping.
- raw/interim quality checks.
- processed panel construction.
- price-volume factor and forward label logic.
- A-share pipeline config snapshots and factor expression parsing.
- evaluation layer contract, artifact schema `0.5`, and `manifest.display`.
- API evidence, CSV preview, direct download, ZIP download, and comparison endpoints.
- React evidence workbench state, run switching, combination switching, and download picker parity.

Do not run slow, network-heavy, destructive, or environment-changing commands without explicit approval.

## 7. Documentation Roles

- [README.md](README.md): current platform entry.
- [ARCHITECTURE.md](ARCHITECTURE.md): platform architecture and module boundaries.
- [EVALUATION_INTERFACE.md](EVALUATION_INTERFACE.md): evaluation layer input/output contract.
- [REPORT.md](REPORT.md): formal research report.
- [RESEARCH_CARD.md](RESEARCH_CARD.md): evidence ledger.
- [PROJECT_STATUS.md](PROJECT_STATUS.md): current local status, next tasks, decisions, and root handoff.
- [docs/architecture/README.md](docs/architecture/README.md): architecture document index and new-session reading order.
- [docs/architecture/handoff-before-directory-rename.md](docs/architecture/handoff-before-directory-rename.md): directory-rename and new-conversation handoff.
- [docs/csmar_data_manual.md](docs/csmar_data_manual.md): CSMAR download, fields, data rules, schema, and private report template.
- [docs/code-map/](docs/code-map/): source reading map.
- [docs/code-map/a-share-interface.md](docs/code-map/a-share-interface.md): A-share build/evaluate entrypoints and frontend snapshot boundary.
- `.codex/skills/quant-research-platform-maintenance/SKILL.md`: project-local maintenance workflow for artifact/API/frontend contract changes.

Root handoff rule: if this project changes its current topic, phase, milestone, blocker, next step, public readiness, or data/API/Git/Conda/privacy boundary, record an `open` signal in [PROJECT_STATUS.md](PROJECT_STATUS.md). Do not edit workspace-root planning or status files from this project unless explicitly asked.

## 8. Forbidden Operations

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
