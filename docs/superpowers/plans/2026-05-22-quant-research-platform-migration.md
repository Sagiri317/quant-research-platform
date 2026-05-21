# Quant Research Platform Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前 `a-share-alpha-lab` 分支迁移为 `quant-research-platform`，完整吸收 `alpha-research-core` 当前实现快照，并让 A 股成为第一个可运行 market module。

**Architecture:** 使用标准 `src/quant_research_platform/` 包结构。原 core 的计算、artifact、API 和前端 workbench 尽量保留实现，改命名空间和产品流程；原 A 股 `code/` 迁入 `markets/a_share/`；前端从 CSV upload 主流程改为 Factor/Pipeline Studio + Evidence Workbench。

**Tech Stack:** Python 3.11、pandas、numpy、pydantic、typer、FastAPI、React、Vite、Vitest、Recharts、Conda `analysis` 环境、Windows PowerShell。

---

## 0. 执行原则

- 当前分支：`codex/a-share-quant-platform`。
- 当前工作目录暂时仍是 `E:\quant-workspace\projects\a-share-alpha-lab`。
- 本计划先在旧目录内完成迁移和测试；目录最终改名为 `quant-research-platform` 是最后单独步骤。
- 不合并 `alpha-research-core` Git 历史，只复制当前文件快照。
- 不删除旧 `alpha-research-core` 项目；用户已批准迁移成功后在旧 core 项目写冻结 / deprecated 说明。真正删除旧项目需要另一个 exact-path 审批。
- 用户已批准旧 `code/` 迁移完成并通过测试后清理，不保留兼容入口。exact path: `E:/quant-workspace/projects/a-share-alpha-lab/code/`。
- 用户已批准两类依赖安装：Conda `analysis` 环境里的 Python 包，以及项目 `web/` 目录下的 npm 依赖。
- 仍然禁止安装到 Conda `base`、使用 `pip`、修改全局 npm/Conda/Git 配置。
- 浏览器验收优先使用本机已安装 Chrome。不要默认安装 Playwright Chromium；只有系统 Chrome 不可用且用户再次确认时，才运行会写用户缓存的 Playwright 浏览器安装。
- 如果缺依赖且属于已批准范围，使用最小命令安装；如果超出范围，停止并报告精确命令。
- 不读取、打印、写入 credentials、`.env`、tokens、cookies。
- 不接外部 API、券商、交易所、paper trading、live trading。
- 不提交 Git commit，除非用户后续明确要求。
- 验证默认使用小样本：一年数据、2-3 个因子、2-3 个 label。先 smoke test，不直接跑 2014-2025 全样本。
- 前端 / 后端服务只在需要浏览器验收时启动；启动前检查端口和已有进程，验收后停止服务，避免占内存。

## 1. 子智能体分工

实现阶段允许超过 5 个子智能体。主线程负责整合、冲突解决和最终验证。

### Agent A: Core Contracts / Engines / Artifacts

**Ownership:**

```text
src/quant_research_platform/core/
src/quant_research_platform/engines/
src/quant_research_platform/artifacts/
tests/core/
tests/engines/
tests/artifacts/
```

**Source:**

```text
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/core/
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/engines/
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/artifacts/
E:/quant-workspace/projects/alpha-research-core/tests/test_metrics.py
E:/quant-workspace/projects/alpha-research-core/tests/test_portfolio.py
E:/quant-workspace/projects/alpha-research-core/tests/test_artifact_contract.py
E:/quant-workspace/projects/alpha-research-core/tests/test_evaluation_set_artifacts.py
```

**Constraints:**

- Preserve schema `0.5`.
- Preserve current formulas unless tests prove a namespace-only bug.
- Do not add legacy schema fallback.
- Do not reintroduce `artifacts.xlsx`.

### Agent B: API / Downloads / Run Repository

**Ownership:**

```text
src/quant_research_platform/api/
tests/api/
```

**Source:**

```text
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/api/
E:/quant-workspace/projects/alpha-research-core/tests/test_api.py
E:/quant-workspace/projects/alpha-research-core/tests/test_run_comparison.py
E:/quant-workspace/projects/alpha-research-core/tests/test_platform_viewer.py
```

**Constraints:**

- Keep local-only FastAPI.
- Keep evidence/download behavior display-manifest-driven.
- Add reserved pipeline/market endpoint structure only if backed by tests or explicit placeholder response.
- Do not create DB storage.

### Agent C: CLI / IO / Package Configuration

**Ownership:**

```text
pyproject.toml
src/quant_research_platform/cli/
src/quant_research_platform/io/
src/quant_research_platform/__init__.py
src/quant_research_platform/_version.py
tests/test_cli.py
tests/test_io.py
```

**Source:**

```text
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/cli/
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/io/
E:/quant-workspace/projects/alpha-research-core/pyproject.toml
```

**Constraints:**

- CLI script name is `qrp`.
- Core command examples use `python -m quant_research_platform.cli`.
- `io/` loads only explicit paths or DataFrames; no sibling-project discovery.
- `pyproject.toml` must not include `../alpha-research-core/src`.

### Agent D: A-Share Market Module

**Ownership:**

```text
src/quant_research_platform/markets/a_share/
tests/markets/a_share/
config/a_share/
```

**Source:**

```text
code/csmar/
code/datasets/
code/factors/
code/evaluation/
code/runtime/
code/validation/
tests/csmar/
tests/datasets/
tests/evaluation/
tests/factors/
tests/runtime/
```

**Constraints:**

- Preserve CSMAR private-data boundaries.
- Preserve `utf-8-sig` behavior for Windows-facing Chinese CSV where relevant.
- Keep DataFrame handoff to core as the primary evaluation path.
- Do not touch private data unless a specific run requires it and the user approves.

### Agent E: Factor Expression / Pipeline Contract

**Ownership:**

```text
src/quant_research_platform/research/
src/quant_research_platform/markets/a_share/factors/expressions.py
src/quant_research_platform/markets/a_share/pipeline/
tests/markets/a_share/test_factor_expressions.py
tests/markets/a_share/test_pipeline_config.py
config/a_share/factor_templates/
config/a_share/pipelines/
```

**Constraints:**

- First version supports whitelist expression parsing and templates.
- No Python `eval`.
- No arbitrary Python code execution.
- Store expression text, parsed structure, field dependencies, template source, and run config snapshot.

### Agent F: Web UI Migration

**Ownership:**

```text
web/
tests/web/ if created
```

**Source:**

```text
E:/quant-workspace/projects/alpha-research-core/web/
```

**Constraints:**

- Keep current visual system and evidence browsing quality.
- Change main flow from CSV upload to Factor/Pipeline Studio.
- CSV upload can remain secondary or developer-only.
- First version enables A-share only; crypto is disabled/planned.
- No frontend business logic that bypasses `manifest.display`.

### Agent G: Docs / AGENTS / Skills / Status

**Ownership:**

```text
README.md
AGENTS.md
PROJECT_STATUS.md
ARCHITECTURE.md
EVALUATION_INTERFACE.md
RESEARCH_CARD.md
REPORT.md
docs/
.codex/skills/
data/README.md
.gitignore
```

**Source:**

```text
Current a-share docs
E:/quant-workspace/projects/alpha-research-core/AGENTS.md
E:/quant-workspace/projects/alpha-research-core/ARCHITECTURE.md
E:/quant-workspace/projects/alpha-research-core/EVALUATION_INTERFACE.md
E:/quant-workspace/projects/alpha-research-core/docs/code-map/
E:/quant-workspace/projects/alpha-research-core/.codex/skills/alpha-research-core-maintenance/
```

**Constraints:**

- Chinese-first user-facing docs.
- Operational files may remain English-first where appropriate.
- `docs/a_share_quant_platform_blueprint.md` is temporary and should be absorbed then removed; user has approved deleting exact path `E:/quant-workspace/projects/a-share-alpha-lab/docs/a_share_quant_platform_blueprint.md` after absorption.
- Root workspace files are not edited in the overnight migration. The user will handle root planning/status sync tomorrow, so this does not block the goal.

## 2. Preflight Snapshot

### Task 0: Dependency Boundary Check

**Files:**

- Read: `pyproject.toml`
- Read: `web/package.json` after web migration exists
- Read: `web/package-lock.json` after web migration exists

- [ ] **Step 1: Check Conda environment**

Run:

```powershell
conda env list
```

Expected: `analysis` exists. Do not use `base` for project work.

- [ ] **Step 2: If Python dependency is missing, install only into analysis**

Allowed pattern:

```powershell
conda install -n analysis <package-name>
```

Use only when the missing package is required by the migrated project and is not
already available in `analysis`.

- [ ] **Step 3: If frontend dependencies are missing, install from web/**

Allowed patterns from project `web/`:

```powershell
npm ci
npm install
```

Prefer `npm ci` when `package-lock.json` is present and unchanged. Do not run
global npm commands.

- [ ] **Step 4: Use installed Chrome for browser acceptance**

Use system Chrome for browser acceptance when possible. Do not run:

```powershell
npx playwright install chromium
```

unless system Chrome cannot be used and the user gives a fresh confirmation.

### Task 1: Baseline Git and File Inventory

**Files:**

- Read: `pyproject.toml`
- Read: `AGENTS.md`
- Read: `docs/platform_migration_design.md`
- Read: `E:/quant-workspace/projects/alpha-research-core/AGENTS.md`
- Read: `E:/quant-workspace/projects/alpha-research-core/.codex/skills/alpha-research-core-maintenance/SKILL.md`

- [ ] **Step 1: Confirm current branch and worktree**

Run:

```powershell
git status --short --branch
```

Expected:

```text
## codex/a-share-quant-platform
```

Known current dirty files before implementation:

```text
M  PROJECT_STATUS.md
M  README.md
?? docs/a_share_quant_platform_blueprint.md
?? docs/platform_migration_design.md
?? docs/superpowers/plans/2026-05-22-quant-research-platform-migration.md
```

- [ ] **Step 2: Confirm old core worktree is not dirty except known blueprint**

Run:

```powershell
git status --short --branch
```

from:

```text
E:/quant-workspace/projects/alpha-research-core
```

Expected known state:

```text
## main
?? docs/core_future_blueprint.md
```

If more files are dirty, stop and inspect before copying.

- [ ] **Step 3: Record source inventories**

Run:

```powershell
Get-ChildItem -LiteralPath 'E:\quant-workspace\projects\alpha-research-core\src\alpha_research_core' -Directory | Select-Object Name
Get-ChildItem -LiteralPath 'E:\quant-workspace\projects\alpha-research-core\tests' -File | Select-Object Name
Get-ChildItem -LiteralPath 'E:\quant-workspace\projects\alpha-research-core\web\src' -File | Select-Object Name
Get-ChildItem -LiteralPath 'code' -Directory | Select-Object Name
Get-ChildItem -LiteralPath 'tests' -Directory | Select-Object Name
```

Expected source domains:

```text
core api artifacts cli engines io runners
csmar datasets evaluation factors runtime validation
```

## 3. Package Skeleton and Configuration

### Task 2: Create New Source Skeleton

**Files:**

- Create: `src/quant_research_platform/__init__.py`
- Create: `src/quant_research_platform/_version.py`
- Create: `src/quant_research_platform/markets/__init__.py`
- Create: `src/quant_research_platform/markets/a_share/__init__.py`
- Create: `src/quant_research_platform/markets/crypto/__init__.py`
- Create: `src/quant_research_platform/research/__init__.py`
- Create: `src/quant_research_platform/runtime/__init__.py`

- [ ] **Step 1: Add package roots**

Create package roots with minimal exports:

```python
"""Local multi-market quantitative research platform."""

from quant_research_platform._version import __version__

__all__ = ["__version__"]
```

`_version.py`:

```python
"""Package version."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Add market namespace docs**

`markets/a_share/__init__.py`:

```python
"""A-share market module backed by local private CSMAR data."""
```

`markets/crypto/__init__.py`:

```python
"""Reserved crypto market module interface."""
```

- [ ] **Step 3: Run import smoke**

Run:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -c "import quant_research_platform as qrp; print(qrp.__version__)"
```

Expected:

```text
0.1.0
```

### Task 3: Update pyproject

**Files:**

- Modify: `pyproject.toml`

- [ ] **Step 1: Update project metadata**

Set:

```toml
[project]
name = "quant-research-platform"
version = "0.1.0"
description = "Local multi-market quantitative research platform."
requires-python = ">=3.11"
dependencies = [
  "numpy",
  "pandas",
  "pyarrow",
  "pydantic",
  "pytest",
  "ruff",
  "typer",
]

[project.scripts]
qrp = "quant_research_platform.cli:main"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

Keep existing ruff settings, and keep `notebooks/` excluded.

- [ ] **Step 2: Verify forbidden pythonpath removed**

Run:

```powershell
Select-String -Path 'pyproject.toml' -Pattern '\.\./alpha-research-core|alpha_research_core'
```

Expected: no output.

## 4. Core Source Migration

### Task 4: Copy Core Calculation and Artifact Modules

**Files:**

- Create/Modify: `src/quant_research_platform/core/`
- Create/Modify: `src/quant_research_platform/engines/`
- Create/Modify: `src/quant_research_platform/artifacts/`
- Test: `tests/core/`
- Test: `tests/engines/`
- Test: `tests/artifacts/`

- [ ] **Step 1: Copy source directories**

Copy current files from:

```text
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/core/
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/engines/
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/artifacts/
```

to:

```text
src/quant_research_platform/core/
src/quant_research_platform/engines/
src/quant_research_platform/artifacts/
```

Do not copy `__pycache__/`.

- [ ] **Step 2: Rewrite imports**

Replace active imports:

```text
alpha_research_core.
```

with:

```text
quant_research_platform.
```

Only leave `alpha_research_core` in migration docs or explicit historical notes.

- [ ] **Step 3: Export core public API**

Ensure `src/quant_research_platform/core/__init__.py` exports at least:

```python
from quant_research_platform.core.contracts import EvaluationSpec, WeightConfig
from quant_research_platform.core.evaluation import run_evaluation

__all__ = ["EvaluationSpec", "WeightConfig", "run_evaluation"]
```

If the copied old core exposes additional public classes used by tests, export
them explicitly rather than relying on wildcard imports.

- [ ] **Step 4: Copy and relocate core tests**

Map old tests:

```text
test_evaluation_engine_contract.py -> tests/core/test_evaluation_engine_contract.py
test_panel.py                      -> tests/core/test_panel.py
test_metrics.py                    -> tests/engines/test_metrics.py
test_portfolio.py                  -> tests/engines/test_portfolio.py
test_neutralization.py             -> tests/core/test_neutralization.py
test_time_breakdowns.py            -> tests/core/test_time_breakdowns.py
test_artifact_contract.py          -> tests/artifacts/test_artifact_contract.py
test_evaluation_set_artifacts.py   -> tests/artifacts/test_evaluation_set_artifacts.py
```

Rewrite imports to `quant_research_platform`.

- [ ] **Step 5: Run narrow tests**

Run:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest tests/core tests/engines tests/artifacts -q
```

Expected: all selected tests pass. If failures are only namespace/import
failures, fix imports. If failures alter metrics behavior, stop and inspect.

## 5. CLI and IO Migration

### Task 5: Migrate CLI and IO

**Files:**

- Create/Modify: `src/quant_research_platform/cli/`
- Create/Modify: `src/quant_research_platform/io/`
- Test: `tests/test_cli.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Copy source**

Copy:

```text
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/cli/
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/io/
```

to:

```text
src/quant_research_platform/cli/
src/quant_research_platform/io/
```

Do not copy `__pycache__/`.

- [ ] **Step 2: Rename CLI identity**

Replace user-facing command names:

```text
alpha-core-demo
alpha_research_core
alpha-research-core
```

with platform wording:

```text
qrp
quant_research_platform
quant-research-platform
```

Keep historical mentions only in migration docs.

- [ ] **Step 3: Add top-level command groups**

`qrp` should expose at minimum:

```text
qrp --help
qrp evaluate-parquet --help
qrp compare --help
qrp serve --help
qrp a-share --help
```

For first migration, `qrp a-share` can route to implemented A-share commands as
they become available, but the help text must not promise crypto support.

- [ ] **Step 4: Narrow IO semantics**

Ensure `io/` only supports explicit CSV/Parquet paths. It must not scan sibling
projects, infer workspace-level project locations, or read arbitrary project
outputs implicitly.

- [ ] **Step 5: Run CLI smoke**

Run:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.cli --help
conda run -n analysis python -m quant_research_platform.cli evaluate-parquet --help
conda run -n analysis python -m quant_research_platform.cli compare --help
conda run -n analysis python -m quant_research_platform.cli a-share --help
```

Expected: each command prints help and exits 0.

## 6. API Migration

### Task 6: Migrate FastAPI Evidence API

**Files:**

- Create/Modify: `src/quant_research_platform/api/`
- Test: `tests/api/`

- [ ] **Step 1: Copy API source**

Copy:

```text
E:/quant-workspace/projects/alpha-research-core/src/alpha_research_core/api/
```

to:

```text
src/quant_research_platform/api/
```

Do not copy `__pycache__/`.

- [ ] **Step 2: Rewrite namespace**

Replace imports and app labels from old core to new platform namespace. API route
behavior for runs, evidence, table preview, direct CSV downloads, ZIP downloads,
and comparison should remain compatible with current schema `0.5`.

- [ ] **Step 3: Add market capability placeholders**

Add route or service behavior that can answer:

```text
GET /markets
GET /markets/a_share/capabilities
GET /markets/crypto/capabilities
```

Expected semantics:

```text
a_share: enabled
crypto: planned / disabled
```

This can be static in the first migration.

- [ ] **Step 4: Add pipeline placeholder models**

Introduce request/response models for pipeline runs without building a database:

```text
PipelineRunRequest
PipelineRunSummary
PipelineCheckpointSummary
```

First version can store and load generated run snapshots from files under
`reports/private_real/pipeline_runs/`.

- [ ] **Step 5: Copy and relocate API tests**

Map old tests:

```text
test_api.py             -> tests/api/test_api.py
test_platform_viewer.py -> tests/api/test_platform_viewer.py
test_run_comparison.py  -> tests/api/test_run_comparison.py
```

Add tests for:

```text
GET /markets includes a_share enabled and crypto disabled
GET /markets/a_share/capabilities returns pipeline_supported true
GET /markets/crypto/capabilities returns pipeline_supported false
```

- [ ] **Step 6: Run API tests**

Run:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest tests/api -q
```

Expected: all API tests pass.

## 7. A-Share Module Migration

### Task 7: Move A-Share Code into Market Module

**Files:**

- Create/Modify: `src/quant_research_platform/markets/a_share/`
- Test: `tests/markets/a_share/`

- [ ] **Step 1: Copy A-share source into target modules**

Copy current source:

```text
code/csmar/       -> src/quant_research_platform/markets/a_share/csmar/
code/datasets/    -> src/quant_research_platform/markets/a_share/datasets/
code/factors/     -> src/quant_research_platform/markets/a_share/factors/
code/evaluation/  -> src/quant_research_platform/markets/a_share/evaluation/
code/validation/  -> src/quant_research_platform/markets/a_share/validation/
```

Split:

```text
code/runtime/
```

into:

```text
src/quant_research_platform/runtime/
src/quant_research_platform/markets/a_share/runtime/
```

Use shared runtime only for market-neutral helpers. Keep A-share private paths
and CSMAR-specific logic under `markets/a_share/`.

- [ ] **Step 2: Rewrite imports**

Replace old imports such as:

```python
from csmar...
from datasets...
from factors...
from evaluation...
from runtime...
from validation...
```

with explicit package imports:

```python
from quant_research_platform.markets.a_share.csmar...
from quant_research_platform.markets.a_share.datasets...
from quant_research_platform.markets.a_share.factors...
from quant_research_platform.markets.a_share.evaluation...
from quant_research_platform.markets.a_share.validation...
from quant_research_platform.runtime...
```

- [ ] **Step 3: Replace core bridge imports**

Replace:

```python
from alpha_research_core...
```

with:

```python
from quant_research_platform...
```

The A-share evaluation adapter should pass DataFrames to `EvaluationSpec` and
`run_evaluation`.

- [ ] **Step 4: Move tests**

Map:

```text
tests/csmar/       -> tests/markets/a_share/csmar/
tests/datasets/    -> tests/markets/a_share/datasets/
tests/evaluation/  -> tests/markets/a_share/evaluation/
tests/factors/     -> tests/markets/a_share/factors/
tests/runtime/     -> tests/markets/a_share/runtime/
```

Rewrite imports to `quant_research_platform`.

- [ ] **Step 5: Run A-share tests**

Run:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest tests/markets/a_share -q
```

Expected: all A-share tests pass without `../alpha-research-core/src`.

## 8. Factor Expression and Pipeline Contract

### Task 8: Add Controlled Factor Expression Parser

**Files:**

- Create: `src/quant_research_platform/markets/a_share/factors/expressions.py`
- Create: `tests/markets/a_share/factors/test_expressions.py`
- Create: `config/a_share/factor_templates/reversal_20d.yaml`
- Create: `config/a_share/factor_templates/momentum_60d.yaml`
- Create: `config/a_share/factor_templates/liquidity_20d.yaml`

- [ ] **Step 1: Write parser tests**

Tests must cover:

```python
def test_parse_allowed_expression_records_dependencies():
    expr = parse_factor_expression("neg(return(close, 20))")
    assert expr.name == "neg"
    assert expr.dependencies == {"close"}


def test_parse_rejects_import():
    with pytest.raises(FactorExpressionError):
        parse_factor_expression("__import__('os').system('dir')")


def test_parse_rejects_unknown_field():
    with pytest.raises(FactorExpressionError):
        parse_factor_expression("return(secret_column, 20)")
```

- [ ] **Step 2: Implement whitelist parser**

Allowed first-version fields:

```text
close
open
high
low
volume
amount
market_cap
industry
```

Allowed first-version functions:

```text
return
mean
std
rank_cs
zscore_cs
neg
neutralize
```

Implementation must parse expressions into a small internal object, not execute
Python code.

- [ ] **Step 3: Add templates**

`reversal_20d.yaml`:

```yaml
id: reversal_20d
name: 20日反转
market: a_share
expression: neg(return(close, 20))
description: 使用20日收盘收益的相反数作为反转候选因子。
```

`momentum_60d.yaml`:

```yaml
id: momentum_60d
name: 60日动量
market: a_share
expression: return(close, 60)
description: 使用60日收盘收益作为动量候选因子。
```

`liquidity_20d.yaml`:

```yaml
id: liquidity_20d
name: 20日流动性
market: a_share
expression: rank_cs(mean(amount, 20))
description: 使用20日成交额均值的截面排名作为流动性候选因子。
```

- [ ] **Step 4: Run expression tests**

Run:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest tests/markets/a_share/factors/test_expressions.py -q
```

Expected: pass.

### Task 9: Add Pipeline Config Snapshot Contract

**Files:**

- Create: `src/quant_research_platform/markets/a_share/pipeline/config.py`
- Create: `src/quant_research_platform/markets/a_share/pipeline/snapshots.py`
- Create: `tests/markets/a_share/pipeline/test_pipeline_config.py`
- Create: `config/a_share/pipelines/sample_reversal_2023_2025.yaml`

- [ ] **Step 1: Define config model**

First-version fields:

```text
market
run_id
date_start
date_end
factor_expression
factor_template_id
portfolio_mode
cost_bps
pnl_return_column
label_columns
output_root
```

Use pydantic if already available through core dependencies.

- [ ] **Step 2: Add blocking/warning classification**

Config validation should classify:

```text
missing market              -> blocking
unsupported market          -> blocking
missing factor_expression   -> blocking
unsupported portfolio_mode  -> blocking
missing optional industry   -> warning when diagnostics request needs it
```

- [ ] **Step 3: Save run snapshot**

Snapshot writer saves:

```text
run_config.yaml
warnings.json
summary.md
```

under:

```text
reports/private_real/pipeline_runs/<run_id>/
```

Do not overwrite existing run directory by default.

- [ ] **Step 4: Add tests**

Tests must prove:

```text
valid config writes snapshot
same run_id creates timestamped or unique directory by default
blocking issue prevents snapshot
warning issue is recorded
```

- [ ] **Step 5: Run pipeline config tests**

Run:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest tests/markets/a_share/pipeline -q
```

Expected: pass.

## 9. Web Migration and Product Flow

### Task 10: Copy Web App and Rename Product Identity

**Files:**

- Create/Modify: `web/`

- [ ] **Step 1: Copy old web app**

Copy:

```text
E:/quant-workspace/projects/alpha-research-core/web/
```

to:

```text
web/
```

Do not copy:

```text
node_modules/
dist/
coverage/
playwright-report/
test-results/
```

- [ ] **Step 2: Rename app identity**

Replace user-facing old names:

```text
Alpha Research Core
alpha-research-core
alpha core
```

with:

```text
Quant Research Platform
Research Evidence Workbench
```

Keep old names only in migration docs.

- [ ] **Step 3: Preserve visual system**

Keep current CSS layout, colors, typography, chart styling, cards, and
download/preview surfaces unless they conflict with the new Factor/Pipeline
workflow.

- [ ] **Step 4: Run current web tests**

Run from `web/`:

```powershell
npm test -- --run --maxWorkers=1
```

Expected: either pass or fail only due to expected renamed labels. Fix renamed
label assertions.

### Task 11: Add Factor/Pipeline Studio UI

**Files:**

- Create: `web/src/FactorStudio.tsx`
- Create: `web/src/PipelineBuilder.tsx`
- Create: `web/src/DataCheckpoints.tsx`
- Modify: `web/src/App.tsx`
- Modify: `web/src/Sidebar.tsx`
- Modify: `web/src/api.ts`
- Test: `web/src/App.test.tsx`
- Test: `web/src/api.test.ts`

- [ ] **Step 1: Add API client methods**

Add client calls:

```text
listMarkets()
listFactorTemplates()
createPipelineRun(request)
getPipelineRunStatus(runId)
getPipelineCheckpoints(runId)
```

- [ ] **Step 2: Add Factor Studio**

UI controls:

```text
Market selector: A-share enabled, Crypto planned/disabled
Template selector: built-in templates
Expression text area: controlled expression input
Dependency preview: fields parsed from expression
Validation panel: parser errors and warnings
```

- [ ] **Step 3: Add Pipeline Builder**

UI controls:

```text
date_start
date_end
portfolio_mode: long_only / long_short
cost_bps
pnl_return_column
label_columns
run_id
output_root display
```

Default: do not serve/open after run.

- [ ] **Step 4: Add Data Checkpoints view**

Show:

```text
run config snapshot path
warnings
interim summary
panel summary
factor summary
evaluation artifact links
```

If no run exists, show an empty state without in-app explanatory tutorial text.

- [ ] **Step 5: Keep Evidence Workbench**

Existing evidence browsing pages must still support:

```text
run switching
combination switching
metric summary
charts
CSV preview
downloads
comparison
report mode
```

- [ ] **Step 6: Run web tests, lint, build**

Run from `web/`:

```powershell
npm test -- --run --maxWorkers=1
npm run lint
npm run build
```

Expected: all pass.

## 10. Documentation and Skill Migration

### Task 12: Update Platform Docs and Skill

**Files:**

- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `PROJECT_STATUS.md`
- Create/Modify: `ARCHITECTURE.md`
- Create/Modify: `EVALUATION_INTERFACE.md`
- Modify: `RESEARCH_CARD.md`
- Modify: `REPORT.md`
- Create/Modify: `docs/code-map/`
- Create/Modify: `.codex/skills/quant-research-platform-maintenance/SKILL.md`
- Delete after absorption: `docs/a_share_quant_platform_blueprint.md`

- [ ] **Step 1: Update README**

README must say:

```text
本项目是本地多市场量化研究平台。
A 股是第一版完整 market module。
core evaluation、artifact、API、workbench 已内置。
crypto/其他市场第一版预留接口。
```

- [ ] **Step 2: Update AGENTS**

AGENTS must include:

```text
project purpose
new src/ package structure
Conda analysis rule
private data boundaries
no external APIs or credentials
no broker/exchange/paper/live trading
deletion exact-path approval
frontend/API/artifact contract rule
subagent policy approved for this migration
```

- [ ] **Step 3: Migrate core maintenance skill**

Copy old core skill:

```text
E:/quant-workspace/projects/alpha-research-core/.codex/skills/alpha-research-core-maintenance/
```

to:

```text
.codex/skills/quant-research-platform-maintenance/
```

Rewrite:

```text
alpha-research-core -> quant-research-platform
alpha_research_core -> quant_research_platform
```

Keep artifact schema `0.5`, `manifest.display`, API/download, React evidence UI
verification requirements.

- [ ] **Step 4: Absorb and delete old A-share-only blueprint**

Move useful content from:

```text
docs/a_share_quant_platform_blueprint.md
```

into:

```text
README.md
ARCHITECTURE.md
PROJECT_STATUS.md
docs/code-map/
```

Then delete exact path:

```text
E:/quant-workspace/projects/a-share-alpha-lab/docs/a_share_quant_platform_blueprint.md
```

This deletion is approved by the user's current direction only for this exact
temporary blueprint file after absorption.

- [ ] **Step 5: Run docs checks**

Run:

```powershell
Select-String -Path 'README.md','AGENTS.md','PROJECT_STATUS.md','ARCHITECTURE.md','EVALUATION_INTERFACE.md','RESEARCH_CARD.md','REPORT.md' -Pattern 'a-share-alpha-lab|alpha-research-core|alpha_research_core|\.\./alpha-research-core'
```

Expected: remaining matches are only historical/migration references, not active
commands or imports.

## 11. Cleanup Old Code Tree

### Task 13: Remove Legacy `code/` After Tests Pass

**Files:**

- Delete exact path after migration and tests pass:

```text
E:/quant-workspace/projects/a-share-alpha-lab/code/
```

**Preconditions:**

```text
src/quant_research_platform/markets/a_share/ exists
A-share tests pass from tests/markets/a_share/
No active import uses old code root
pyproject.toml pythonpath is only ["src"]
```

User has pre-approved cleanup of this exact path after the preconditions pass:

```text
E:/quant-workspace/projects/a-share-alpha-lab/code/
```

- [ ] **Step 1: Search active references**

Run:

```powershell
Select-String -Path 'pyproject.toml','tests\**\*.py','src\**\*.py' -Pattern 'pythonpath = \["code"|from csmar|from datasets|from factors|from evaluation|from runtime|from validation'
```

Expected: no active old-root imports.

- [ ] **Step 2: Remove old `code/`**

Use PowerShell native removal only after preconditions pass. This exact path is
pre-approved by the user for post-test cleanup:

```text
E:/quant-workspace/projects/a-share-alpha-lab/code/
```

Do not delete any other directory. If a tool or Windows lock prevents cleanup,
leave `code/` in place and report the exact path for manual deletion.

- [ ] **Step 3: Verify cleanup**

Run:

```powershell
Test-Path -LiteralPath 'code'
git status --short
```

Expected: `code` path absent or explicitly reported as pending manual cleanup.

## 12. Full Verification

### Task 14: Run Python Verification

**Files:**

- All migrated Python source and tests.

- [ ] **Step 1: Run full Python tests**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff**

Run:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m ruff check src tests
```

Expected: pass.

- [ ] **Step 3: Run CLI smoke**

Run:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.cli --help
conda run -n analysis python -m quant_research_platform.cli a-share --help
```

Expected: help text exits 0 and refers to `qrp` / Quant Research Platform.

### Task 14A: Run Small A-Share Pipeline Smoke

**Files:**

- Read/write only approved generated outputs under `reports/private_real/`.
- Use private processed data only if the implementation task explicitly needs
  the smoke run and the data path is already part of the project contract.

- [ ] **Step 1: Select small smoke scope**

Use:

```text
date range: one calendar year
factor count: 2-3
label count: 2-3
portfolio modes: start with long_only
cost settings: one default cost value
```

Recommended first smoke:

```text
2024-01-01 to 2024-12-31
factors: rev_20d, rev_60d, liquidity_20d or closest available existing names
labels: forward_return_1d, forward_return_5d, forward_return_20d if available
```

- [ ] **Step 2: Run smoke through new platform path**

Use the migrated CLI or a checked-in smoke test. Example shape:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.cli a-share run-pipeline --config config/a_share/pipelines/smoke_2024.yaml
```

If the exact CLI command is not ready, run the narrowest pytest that exercises
the same path.

- [ ] **Step 3: Verify smoke outputs**

Check:

```text
run_config.yaml exists
warnings.json exists
evaluation manifest exists
manifest.display exists
row counts are nonzero
no unsupported schema
no full-sample 2014-2025 run was triggered
```

Do not expand to multi-year or full-sample runs during the overnight goal unless
the one-year smoke passes and memory/resource usage is clearly safe.

### Task 15: Run Frontend Verification

**Files:**

- `web/`

- [ ] **Step 1: Run frontend tests**

Run from `web/`:

```powershell
npm test -- --run --maxWorkers=1
```

Expected: pass.

- [ ] **Step 2: Run lint**

Run from `web/`:

```powershell
npm run lint
```

Expected: pass.

- [ ] **Step 3: Run build**

Run from `web/`:

```powershell
npm run build
```

Expected: pass.

### Task 16: Run Contract Searches

**Files:**

- Whole project, excluding `.git`, caches, private data, generated heavy outputs.

- [ ] **Step 1: Search old package path**

Run:

```powershell
Get-ChildItem -Recurse -File | Where-Object {
  $_.FullName -notmatch '\\.git\\' -and
  $_.FullName -notmatch '\\.pytest_cache\\' -and
  $_.FullName -notmatch '\\.ruff_cache\\' -and
  $_.FullName -notmatch '\\data\\raw_private\\' -and
  $_.FullName -notmatch '\\data\\interim_private\\' -and
  $_.FullName -notmatch '\\data\\processed_private\\' -and
  $_.FullName -notmatch '\\reports\\private_real\\'
} | Select-String -Pattern 'alpha_research_core|\.\./alpha-research-core'
```

Expected: only migration docs or historical references remain.

- [ ] **Step 2: Search retired artifact terms**

Run:

```powershell
Get-ChildItem -Recurse -File src,tests,web,docs | Select-String -Pattern 'artifacts.xlsx|total_.*_portfolio_return'
```

Expected: no active code or test references.

- [ ] **Step 3: Search old product names**

Run:

```powershell
Get-ChildItem -Recurse -File src,tests,web,docs | Select-String -Pattern 'Alpha Research Core|alpha-research-core'
```

Expected: only migration docs or explicit historical explanations remain.

## 13. Optional Browser Acceptance

### Task 17: Browser Acceptance Pass

**Precondition:** API and Vite dependencies already available. Use installed
Chrome for browser acceptance where possible. Keep only one API and one Vite
service running.

- [ ] **Step 0: Check existing services and ports**

Before starting services, check likely ports:

```powershell
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in 5173,5174,8000,8001 } | Select-Object LocalAddress,LocalPort,OwningProcess
```

If a previous API or Vite process is already running for this project, reuse it
only if it points to the migrated project. Otherwise stop or choose a different
port and record it.

- [ ] **Step 1: Start API service**

Use a local command consistent with migrated app entrypoint, for example:

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.api.app
```

If the API app uses uvicorn command entry, use the migrated documented command.

- [ ] **Step 2: Start Vite service**

Run from `web/`:

```powershell
npm run dev
```

If port 5173 is occupied, use a different Vite port and record it.

- [ ] **Step 3: Verify manually or with Playwright**

Check:

```text
Research Runs loads
Factor Studio shows A-share enabled and Crypto disabled/planned
Expression template fill works
Pipeline Builder validates required fields
Evidence Workbench shows run list and evidence surfaces
run switching clears stale state
combination switching updates charts, summaries, downloads, CSV preview
downloads do not fail
narrow viewport has no unusable overlap
```

- [ ] **Step 4: Stop services**

Stop local API and Vite processes. Do not delete generated runs or dependencies.

Verify stop:

```powershell
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in 5173,5174,8000,8001 } | Select-Object LocalAddress,LocalPort,OwningProcess
```

## 14. Directory Rename and Old Core Freeze

### Task 18: Final Local Directory Rename Is User-Owned

**Preconditions:**

```text
all tests pass
docs updated
old code/ removed or explicitly marked pending manual cleanup
no active references to old project path
```

- [ ] **Step 1: Record rename handoff**

Do not rename the project directory during the overnight goal. The user will
rename it manually after reviewing the result. Final report should state:

```text
Recommended manual rename:
E:/quant-workspace/projects/a-share-alpha-lab
-> E:/quant-workspace/projects/quant-research-platform
```

### Task 19: Freeze Old Core Project

**Files:**

- Do not delete old core project in this migration.
- User has approved writing a deprecated/frozen note in the old core project.

Recommended outcome:

```text
E:/quant-workspace/projects/alpha-research-core remains as archived reference.
New platform is the active implementation.
Deletion of old core requires a later exact-path approval.
```

Suggested old-core note location:

```text
E:/quant-workspace/projects/alpha-research-core/PROJECT_STATUS.md
```

or a small new doc:

```text
E:/quant-workspace/projects/alpha-research-core/docs/deprecated_after_platform_migration.md
```

Keep the note short and do not alter core code.

## 15. Final Report Requirements

Final implementation report must include:

```text
changed files summary
old code/ cleanup status
old core freeze status
tests run and results
frontend checks run and results
browser acceptance status
remaining risks
next practical step
```

If a test was not run, state why. If dependency installation is needed, do not
install; provide the exact Conda or npm command for user approval.

For this migration, update that rule with the user's standing approval:

```text
Conda installs into analysis are approved.
npm installs from project web/ are approved.
Anything outside those boundaries still needs explicit approval.
```

## 16. Future Feature Triage

The file `C:/Users/DELL/Downloads/core_future_features_summary.md` is treated as
a roadmap input, not as a requirement to implement everything in the migration.

### Keep or carry over in this migration

These are already implemented or close to existing core surfaces and should be
preserved while renaming:

```text
IC / RankIC
grouped return
factor neutralization if already present
long-only
long-short
transaction cost
Parquet explicit input
multi-market interface skeleton
label / PnL return contract
evidence workbench
```

### Lightweight hooks allowed in this migration

These may be represented as interfaces, config fields, or simple summaries, but
must not expand into full new subsystems during migration:

```text
multi-factor composite columns as explicit inputs
risk exposure columns as explicit inputs
liquidity / capacity summary placeholders
regime split by provided column
walk-forward interface placeholders
Tail Risk as optional simple performance diagnostic
```

### Separate post-migration projects

These require their own design and test plan:

```text
factor orthogonalization
automatic multi-factor search
covariance matrix
Barra-like risk model
risk attribution
beta neutral
optimizer
risk parity
slippage
market impact
XGBoost / LightGBM
ensemble model
regime detection
multi-frequency full support
tick data support
website deployment
```

Website deployment must be planned separately because it changes local-only,
privacy, API exposure, authentication, and hosting boundaries. The migration can
make the web app buildable, but should not deploy it.

## 17. Self-Review Checklist for This Plan

- [ ] Design sections 1-19 are covered by tasks.
- [ ] No plan step depends on `../alpha-research-core/src` after migration.
- [ ] Core artifact contract and frontend evidence contract have explicit tests.
- [ ] A-share private data boundaries remain explicit.
- [ ] Frontend main flow changes from CSV upload to Factor/Pipeline Studio.
- [ ] Crypto is placeholder only.
- [ ] Old `code/` cleanup is gated after tests.
- [ ] Old core deletion is excluded.
- [ ] Verification commands are PowerShell-safe.
