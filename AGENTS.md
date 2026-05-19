# AGENTS.md

Rules for Codex work in `a-share-alpha-lab`.

This file is for safety and execution boundaries only. Project narrative belongs in [README.md](README.md), [REPORT.md](REPORT.md), [RESEARCH_CARD.md](RESEARCH_CARD.md), and [PROJECT_STATUS.md](PROJECT_STATUS.md).

## 1. Scope

Project purpose:

- Local private CSMAR A-share factor research.
- Code and public-safe aggregate summaries may later be published after review.
- CSMAR raw/interim/processed data and private real reports remain private.

Current structure:

- `code/csmar/`: CSMAR raw table contracts, metadata, ingestion, and interim quality.
- `code/datasets/`: interim tables to full research panel.
- `code/factors/`: full panel to slim factor table and forward labels.
- `code/evaluation/`: `alpha-research-core` bridge and summary helpers.
- `code/validation/` and `code/runtime/`: quality checks, space gates, and shared runtime helpers.

Hard boundaries:

- Do not publish, commit, copy, or expose CSMAR raw/interim/processed data.
- Do not log in to CSMAR, automate browser downloads, use CSMAR API, or handle credentials.
- Do not create, read, print, store, or commit credentials, tokens, secrets, or `.env` files.
- Do not describe this project as production automation, paper trading, live trading, or investment advice.
- Do not claim real market performance, factor validity, benchmark performance, or universe coverage from preliminary diagnostics.
- Do not edit sibling projects or workspace-root files unless explicitly asked.

## 2. Editable Area

Project-level Codex may edit:

```text
README.md
REPORT.md
RESEARCH_CARD.md
PROJECT_STATUS.md
AGENTS.md
data/README.md
docs/
code/
tests/
notebooks/
reports/
pyproject.toml
```

Private data directories may be read or transformed only when the user has approved that task:

```text
data/raw_private/
data/interim_private/
data/processed_private/
reports/private_real/
```

Do not delete files or directories unless the user explicitly approves the exact path.

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
- Use only Conda for dependency management.
- Do not use `pip` or `python -m pip`.
- Do not install, update, or remove dependencies without explicit user approval.
- Do not create additional environments without explicit user approval.

Common checks:

```powershell
conda run -n analysis python -m pytest
conda run -n analysis python -m ruff check .
```

On Windows, `conda run` can emit OpenCL vendor noise even when the script succeeds. Use exit codes and test summaries, not the noise line alone, to judge success.

## 5. Data Responsibilities

CSMAR data layers:

```text
data/raw_private/csmar/          original CSMAR exports
data/interim_private/csmar/      normalized audit layer
data/processed_private/csmar/    research-ready Parquet
reports/private_real/            private quality and evaluation reports
```

Current processed contract:

- `panel_full/research_panel_<start>_<end>.parquet` is the full-sample research panel.
- `price_volume_factors_full/price_volume_factors_<start>_<end>.parquet` is the slim factor and label table.
- Slim factor files must keep `stock_code`, `trade_date`, tradability columns, common liquidity/market-cap columns, factor columns, and forward label columns.

`alpha-research-core` integration:

- A-share produces processed Parquet and adapts fields.
- Core receives an in-memory DataFrame through `EvaluationSpec` and `run_evaluation(spec)`.
- Do not modify `alpha-research-core` from this project session unless the user explicitly asks.

## 6. Testing

Run the narrowest relevant tests first.

Important test surfaces:

- CSMAR field contracts, filename parsing, metadata, and field mapping.
- raw/interim quality checks.
- processed panel construction.
- price-volume factor and forward label logic.
- slim factor output and space gate logic.
- core evaluation bridge.

Do not run slow, network-heavy, destructive, or environment-changing commands without explicit approval.

## 7. Documentation Roles

- [README.md](README.md): current project entry.
- [REPORT.md](REPORT.md): formal research report.
- [RESEARCH_CARD.md](RESEARCH_CARD.md): evidence ledger.
- [PROJECT_STATUS.md](PROJECT_STATUS.md): current local status, next tasks, decisions, and root handoff.
- [docs/csmar_data_manual.md](docs/csmar_data_manual.md): CSMAR download, fields, data rules, schema, and private report template.
- [docs/code-map/](docs/code-map/): source reading map, including `alpha-research-core` handoff details.

Root handoff rule: if this project changes its current topic, phase, milestone, blocker, next step, public readiness, or data/API/Git/Conda/privacy boundary, record an `open` signal in [PROJECT_STATUS.md](PROJECT_STATUS.md). Do not edit workspace-root planning or status files from this project unless explicitly asked.

## 8. Forbidden Operations

Do not:

- Delete files or directories without exact explicit approval.
- Use recursive or wildcard deletion commands without explicit approval.
- Use unapproved external data.
- Connect vendor APIs, exchange APIs, broker APIs, or web data APIs.
- Create, read, print, or store credentials.
- Install, update, or remove dependencies without approval.
- Use `pip` in this project.
- Hardcode absolute paths in reusable files.
- Copy private notes, copyrighted materials, full problem banks, or copied solutions.
- Build downloader automation, trading execution, paper trading, live trading, order routing, or production portfolio automation.
