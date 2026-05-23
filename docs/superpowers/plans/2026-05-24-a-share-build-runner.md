# A-Share Build Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full A-share local pipeline runner that can orchestrate interim build, panel build, factor build, and evaluation while preserving the existing `processed_existing` lightweight runner.

**Architecture:** Keep `run_processed_existing_pipeline()` as the Phase 1 lightweight path and add a separate full-build runner in `src/quant_research_platform/pipelines/a_share/`. The full runner should execute named steps through small wrappers, write one checkpoint per step, sanitize failures, avoid overwriting run directories by default, and expose status through the existing Pipeline Builder API without bypassing artifact/display contracts.

**Tech Stack:** Python 3.11, pandas/pyarrow, FastAPI/Pydantic, pytest, Ruff, PowerShell on Windows, Conda environment `analysis`.

---

## Scope

Implement N1 only:

- Add a full A-share build/evaluate execution mode.
- Orchestrate existing local modules:
  - `quant_research_platform.markets.implementations.a_share.csmar.build_interim`
  - `quant_research_platform.markets.implementations.a_share.datasets.build_panel`
  - `quant_research_platform.markets.implementations.a_share.factors.build_factors`
  - `quant_research_platform.markets.implementations.a_share.evaluation.evaluation_adapter.run_processed_factor_evaluation`
- Write safe runtime summaries and checkpoints for each step.
- Add narrow tests with temporary paths and monkeypatched step functions.
- Update docs that describe the pipeline boundary.

Do not:

- Download CSMAR or automate browser/API login.
- Read, print, commit, or copy private raw/interim/processed data.
- Delete generated runs.
- Replace `processed_existing`.
- Implement N2 tradability diagnostics, N3 ZIP packaging, or N4 frontend feature split in this branch.

## Planned Files

- Modify `src/quant_research_platform/pipelines/a_share/config.py`
  - Add full-build configuration fields needed by orchestration while keeping existing defaults compatible.
- Modify `src/quant_research_platform/pipelines/a_share/pipeline_step_registry.py`
  - Register `build_interim`, `build_panel`, `build_factors`, and `evaluate` step contracts.
- Create `src/quant_research_platform/pipelines/a_share/full_build_runner.py`
  - Own full-build orchestration, step wrappers, checkpoint writing, runtime summary writing, failure sanitization, and result object.
- Modify `src/quant_research_platform/pipelines/a_share/runner.py`
  - Keep the processed-existing runner intact; optionally share checkpoint payload helpers only if the diff remains small and obvious.
- Modify `src/quant_research_platform/interfaces/api/services/pipeline_service.py`
  - Accept `execution_mode="full_build"` and call the full runner.
- Modify `src/quant_research_platform/interfaces/api/routes/pipeline.py`
  - No route shape change expected; update docstrings only if needed.
- Modify `tests/pipelines/a_share/test_pipeline_config.py`
  - Validate new config fields and unsafe path/run-id behavior.
- Create `tests/pipelines/a_share/test_full_build_runner.py`
  - Test successful full build, skipped interim mode, failure checkpoint, overwrite behavior, and sanitized failure messages.
- Modify `tests/pipelines/a_share/test_pipeline_runner.py`
  - Keep existing processed-existing tests passing; add registry assertions only if not covered in the new test file.
- Modify API tests if an existing pipeline API test file exists; otherwise add focused coverage in `tests/pipelines/a_share/test_full_build_runner.py` and defer route-level coverage to a separate API task.
- Modify `docs/architecture/next-stage-roadmap.md`
  - Mark N1 as the current implementation target only after code is complete.
- Modify `PROJECT_STATUS.md`
  - Update root handoff only after the branch changes current project status, not during the first code-only checkpoint.

## Design Notes

- Use a new `full_build_runner.py` instead of growing `runner.py`; `runner.py` already represents the lightweight processed-existing path.
- Prefer direct Python function calls over subprocess shell commands. If an existing module only exposes `main(argv)`, call `main([...])` with explicit arguments and assert return code `0`.
- Step outputs should be relative paths under the run directory when possible. For private data outputs outside the run directory, record only logical artifact labels and sanitized row counts, not absolute private paths.
- Failure messages stored in `runtime_summary.json` and `checkpoints.json` must use stable failure codes, not raw exception text that may include private paths or column names from private data.
- `source_alias="pipeline"` remains the API-visible source for evaluation runs produced by the pipeline.

## Task 1: Baseline And Worktree Preparation

**Files:**
- Read: `AGENTS.md`
- Read: `PROJECT_STATUS.md`
- Read: `docs/architecture/next-stage-worktree-plan.md`
- Read: `src/quant_research_platform/pipelines/a_share/runner.py`
- Read: `src/quant_research_platform/interfaces/api/services/pipeline_service.py`

- [ ] **Step 1: Confirm clean checkout**

Run:

```powershell
git status --short
git branch --show-current
git worktree list
```

Expected: no modified files in the target worktree, current branch is `codex/a-share-build-runner` once execution starts.

- [ ] **Step 2: Confirm Python import baseline**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share
```

Expected: existing pipeline tests pass before implementation.

## Task 2: Register Full Build Step Contracts

**Files:**
- Modify: `src/quant_research_platform/pipelines/a_share/pipeline_step_registry.py`
- Test: `tests/pipelines/a_share/test_full_build_runner.py`

- [ ] **Step 1: Write failing registry test**

Create `tests/pipelines/a_share/test_full_build_runner.py` with:

```python
from quant_research_platform.pipelines.a_share.pipeline_step_registry import (
    get_pipeline_step_contract,
)


def test_full_build_step_registry_declares_ordered_steps() -> None:
    build_interim = get_pipeline_step_contract("build_interim")
    build_panel = get_pipeline_step_contract("build_panel")
    build_factors = get_pipeline_step_contract("build_factors")
    evaluate = get_pipeline_step_contract("evaluate")

    assert build_interim.outputs == ("data/interim_private/csmar",)
    assert "raw_private_csmar" in build_interim.inputs
    assert build_panel.inputs == ("data/interim_private/csmar",)
    assert build_panel.outputs == ("research_panel",)
    assert build_factors.inputs == ("research_panel",)
    assert build_factors.outputs == ("price_volume_factor_table",)
    assert evaluate.inputs == ("price_volume_factor_table", "PipelineConfig")
    assert "checkpoint" in evaluate.capabilities
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_full_build_runner.py::test_full_build_step_registry_declares_ordered_steps
```

Expected: FAIL with `unsupported pipeline step: 'build_interim'`.

- [ ] **Step 3: Add contracts**

Modify `PIPELINE_STEP_CONTRACTS`:

```python
    "build_interim": PipelineStepContract(
        name="build_interim",
        version="1",
        inputs=("raw_private_csmar",),
        outputs=("data/interim_private/csmar",),
        capabilities=("build", "checkpoint", "private_data_boundary"),
    ),
    "build_panel": PipelineStepContract(
        name="build_panel",
        version="1",
        inputs=("data/interim_private/csmar",),
        outputs=("research_panel",),
        capabilities=("build", "checkpoint"),
    ),
    "build_factors": PipelineStepContract(
        name="build_factors",
        version="1",
        inputs=("research_panel",),
        outputs=("price_volume_factor_table",),
        capabilities=("build", "checkpoint"),
    ),
    "evaluate": PipelineStepContract(
        name="evaluate",
        version="1",
        inputs=("price_volume_factor_table", "PipelineConfig"),
        outputs=(
            "evaluation/evaluation_manifest.json",
            "evaluation/label_availability.csv",
            "evaluation/evaluation_runs/",
        ),
        capabilities=("evaluate", "checkpoint"),
    ),
```

Keep the existing `snapshot` and `processed_evaluation` entries unchanged.

- [ ] **Step 4: Run registry test**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_full_build_runner.py::test_full_build_step_registry_declares_ordered_steps
```

Expected: PASS.

## Task 3: Add Full Build Config Fields

**Files:**
- Modify: `src/quant_research_platform/pipelines/a_share/config.py`
- Test: `tests/pipelines/a_share/test_pipeline_config.py`

- [ ] **Step 1: Write failing config test**

Append to `tests/pipelines/a_share/test_pipeline_config.py`:

```python
def test_full_build_config_fields_are_serialized(tmp_path):
    config = _valid_config(tmp_path)
    config.raw_root = tmp_path / "raw"
    config.interim_root = tmp_path / "interim"
    config.processed_root = tmp_path / "processed"
    config.rebuild_interim = True
    config.rebuild_panel = True
    config.rebuild_factors = True
    config.min_listing_trading_days = 90
    config.factor_windows = (20, 60)
    config.label_horizons = (1, 20)

    payload = config.to_snapshot_dict()

    assert payload["raw_root"].endswith("/raw")
    assert payload["interim_root"].endswith("/interim")
    assert payload["processed_root"].endswith("/processed")
    assert payload["rebuild_interim"] is True
    assert payload["rebuild_panel"] is True
    assert payload["rebuild_factors"] is True
    assert payload["min_listing_trading_days"] == 90
    assert payload["factor_windows"] == [20, 60]
    assert payload["label_horizons"] == [1, 20]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_pipeline_config.py::test_full_build_config_fields_are_serialized
```

Expected: FAIL with `AttributeError` for `raw_root` or missing serialized keys.

- [ ] **Step 3: Extend `PipelineConfig`**

Add fields to `PipelineConfig` after `output_root`:

```python
    raw_root: Path = Path("data/raw_private/csmar")
    interim_root: Path = Path("data/interim_private/csmar")
    processed_root: Path = Path("data/processed_private/csmar")
    rebuild_interim: bool = False
    rebuild_panel: bool = False
    rebuild_factors: bool = False
    min_listing_trading_days: int = 120
    factor_windows: tuple[int, ...] = (20, 60, 120)
    label_horizons: tuple[int, ...] = (1, 5, 10, 20, 60, 120)
```

Update `from_mapping()`:

```python
        payload["raw_root"] = Path(payload.get("raw_root", "data/raw_private/csmar"))
        payload["interim_root"] = Path(
            payload.get("interim_root", "data/interim_private/csmar")
        )
        payload["processed_root"] = Path(
            payload.get("processed_root", "data/processed_private/csmar")
        )
        payload["factor_windows"] = tuple(payload.get("factor_windows", (20, 60, 120)))
        payload["label_horizons"] = tuple(
            payload.get("label_horizons", (1, 5, 10, 20, 60, 120))
        )
```

Update `to_snapshot_dict()`:

```python
        payload["raw_root"] = self.raw_root.as_posix()
        payload["interim_root"] = self.interim_root.as_posix()
        payload["processed_root"] = self.processed_root.as_posix()
        payload["factor_windows"] = list(self.factor_windows)
        payload["label_horizons"] = list(self.label_horizons)
```

- [ ] **Step 4: Add validation for numeric fields**

Append to `validate_pipeline_config()`:

```python
    if config.min_listing_trading_days < 0:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="negative_min_listing_trading_days",
                message="min_listing_trading_days must be nonnegative",
                field="min_listing_trading_days",
            )
        )
    if not config.factor_windows or any(window <= 0 for window in config.factor_windows):
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="invalid_factor_windows",
                message="factor_windows must contain positive integers",
                field="factor_windows",
            )
        )
    if not config.label_horizons or any(horizon <= 0 for horizon in config.label_horizons):
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="invalid_label_horizons",
                message="label_horizons must contain positive integers",
                field="label_horizons",
            )
        )
```

- [ ] **Step 5: Run config tests**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_pipeline_config.py
```

Expected: PASS.

## Task 4: Create Full Build Runner Skeleton

**Files:**
- Create: `src/quant_research_platform/pipelines/a_share/full_build_runner.py`
- Test: `tests/pipelines/a_share/test_full_build_runner.py`

- [ ] **Step 1: Write failing import and result test**

Append:

```python
from pathlib import Path

from quant_research_platform.pipelines.a_share.config import PipelineConfig
from quant_research_platform.pipelines.a_share.full_build_runner import (
    FullBuildRunResult,
    run_full_build_pipeline,
)


def _full_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        market="a_share",
        run_id="full_run",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="rev_20d",
        factor_template_id="custom",
        portfolio_mode="long_only",
        cost_bps=0.0,
        pnl_return_column="forward_return_1d",
        label_columns=("forward_return_1d",),
        engine="dataframe",
        output_root=tmp_path / "runs",
        raw_root=tmp_path / "raw",
        interim_root=tmp_path / "interim",
        processed_root=tmp_path / "processed",
        factor_windows=(20,),
        label_horizons=(1,),
    )


def test_full_build_runner_returns_result_type(tmp_path, monkeypatch) -> None:
    config = _full_config(tmp_path)

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_build_interim_step",
        lambda config: {"row_count": 1, "artifact": "interim"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_build_panel_step",
        lambda config: {"row_count": 2, "artifact": "panel"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_build_factors_step",
        lambda config: {"row_count": 3, "artifact": "factors"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_evaluate_step",
        lambda config, run_dir: {
            "row_count": 4,
            "artifact": "evaluation/evaluation_manifest.json",
            "evaluation_manifest_path": run_dir / "evaluation/evaluation_manifest.json",
        },
    )

    result = run_full_build_pipeline(config)

    assert isinstance(result, FullBuildRunResult)
    assert result.status == "ready"
    assert [checkpoint.key for checkpoint in result.checkpoints] == [
        "snapshot",
        "build_interim",
        "build_panel",
        "build_factors",
        "evaluate",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_full_build_runner.py::test_full_build_runner_returns_result_type
```

Expected: FAIL with `ModuleNotFoundError` for `full_build_runner`.

- [ ] **Step 3: Add skeleton module**

Create `full_build_runner.py`:

```python
"""Run the full local A-share build/evaluate pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quant_research_platform.pipelines.a_share.config import (
    PipelineConfig,
    validate_pipeline_config,
)
from quant_research_platform.pipelines.a_share.runner import (
    PipelineCheckpoint,
    PipelineRunnerError,
)
from quant_research_platform.pipelines.a_share.snapshots import (
    PipelineSnapshotError,
    write_run_snapshot,
)


@dataclass(frozen=True)
class FullBuildRunResult:
    run_id: str
    status: str
    run_dir: Path
    checkpoints: list[PipelineCheckpoint]
    runtime_summary_path: Path
    evaluation_manifest_path: Path | None = None
    warnings: list[dict[str, object]] = field(default_factory=list)
    failed_step: str | None = None


def run_full_build_pipeline(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
) -> FullBuildRunResult:
    started = time.perf_counter()
    validation = validate_pipeline_config(config, allow_processed_factor_expression=True)
    if validation.has_blocking:
        codes = ", ".join(issue.code for issue in validation.blocking)
        raise PipelineRunnerError(
            f"blocking config issues prevent full pipeline execution: {codes}"
        )
    try:
        snapshot = write_run_snapshot(
            config,
            overwrite=overwrite,
            allow_processed_factor_expression=True,
        )
    except PipelineSnapshotError as exc:
        raise PipelineRunnerError(str(exc)) from exc

    warnings = [asdict(issue) for issue in validation.warnings]
    checkpoints = [
        PipelineCheckpoint(
            key="snapshot",
            label="配置快照",
            status="ready",
            artifact="run_config.yaml",
            warnings=warnings,
        )
    ]
    evaluation_manifest_path: Path | None = None
    for key, label, function in (
        ("build_interim", "A 股 interim 构建", _run_build_interim_step),
        ("build_panel", "A 股 research panel 构建", _run_build_panel_step),
        ("build_factors", "A 股 factor table 构建", _run_build_factors_step),
        ("evaluate", "A 股 evaluation", _run_evaluate_step),
    ):
        step_started = time.perf_counter()
        try:
            if key == "evaluate":
                payload = function(config, snapshot.run_dir)
            else:
                payload = function(config)
        except Exception as exc:
            failure_code = _failure_code_from_exception(exc)
            checkpoints.append(
                PipelineCheckpoint(
                    key=key,
                    label=label,
                    status="failed",
                    message=f"{key} failed",
                    duration_seconds=_elapsed(step_started),
                    failure_code=failure_code,
                )
            )
            summary_path = _write_runtime_summary(
                snapshot.run_dir,
                config=config,
                status="failed",
                checkpoints=checkpoints,
                warnings=warnings,
                failed_step=key,
                failure_code=failure_code,
                started=started,
            )
            _write_checkpoints(snapshot.run_dir, checkpoints)
            raise PipelineRunnerError(
                f"full pipeline failed at {key}: {failure_code}",
                run_dir=snapshot.run_dir,
                failure_code=failure_code,
            ) from exc
        artifact = payload.get("artifact")
        row_count = payload.get("row_count")
        if key == "evaluate":
            value = payload.get("evaluation_manifest_path")
            if isinstance(value, Path):
                evaluation_manifest_path = value
        checkpoints.append(
            PipelineCheckpoint(
                key=key,
                label=label,
                status="ready",
                artifact=str(artifact) if artifact is not None else None,
                duration_seconds=_elapsed(step_started),
                row_count=int(row_count) if row_count is not None else None,
            )
        )

    summary_path = _write_runtime_summary(
        snapshot.run_dir,
        config=config,
        status="ready",
        checkpoints=checkpoints,
        warnings=warnings,
        failed_step=None,
        failure_code=None,
        started=started,
    )
    _write_checkpoints(snapshot.run_dir, checkpoints)
    return FullBuildRunResult(
        run_id=snapshot.run_dir.name,
        status="ready",
        run_dir=snapshot.run_dir,
        checkpoints=checkpoints,
        runtime_summary_path=summary_path,
        evaluation_manifest_path=evaluation_manifest_path,
        warnings=warnings,
    )
```

Also add local helper stubs in the same file:

```python
def _run_build_interim_step(config: PipelineConfig) -> dict[str, Any]:
    raise NotImplementedError("build_interim step is not wired yet")


def _run_build_panel_step(config: PipelineConfig) -> dict[str, Any]:
    raise NotImplementedError("build_panel step is not wired yet")


def _run_build_factors_step(config: PipelineConfig) -> dict[str, Any]:
    raise NotImplementedError("build_factors step is not wired yet")


def _run_evaluate_step(config: PipelineConfig, run_dir: Path) -> dict[str, Any]:
    raise NotImplementedError("evaluate step is not wired yet")
```

Add copied helper implementations from `runner.py` for `_elapsed()`, `_write_checkpoints()`, and `_failure_code_from_exception()`. If duplication becomes noisy, refactor them into `pipeline_runtime.py` in a later task.

- [ ] **Step 4: Run skeleton test**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_full_build_runner.py::test_full_build_runner_returns_result_type
```

Expected: PASS.

## Task 5: Wire Step Wrappers To Existing A-Share Modules

**Files:**
- Modify: `src/quant_research_platform/pipelines/a_share/full_build_runner.py`
- Test: `tests/pipelines/a_share/test_full_build_runner.py`

- [ ] **Step 1: Write wrapper argument test**

Append:

```python
def test_full_build_step_wrappers_pass_expected_arguments(tmp_path, monkeypatch) -> None:
    config = _full_config(tmp_path)
    calls: list[tuple[str, list[str]]] = []

    def fake_build_interim(argv):
        calls.append(("interim", argv))
        return 0

    def fake_build_panel(argv):
        calls.append(("panel", argv))
        return 0

    def fake_build_factors(argv):
        calls.append(("factors", argv))
        return 0

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "build_interim_main",
        fake_build_interim,
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "build_panel_main",
        fake_build_panel,
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "build_factors_main",
        fake_build_factors,
    )

    assert _run_build_interim_step(config)["artifact"] == "interim_private/csmar"
    assert _run_build_panel_step(config)["artifact"] == "processed_private/csmar/panel_full"
    assert _run_build_factors_step(config)["artifact"] == (
        "processed_private/csmar/price_volume_factors_full"
    )

    assert calls[0] == (
        "interim",
        [
            "--raw-root",
            config.raw_root.as_posix(),
            "--output-root",
            config.interim_root.as_posix(),
            "--start-date",
            config.date_start,
            "--end-date",
            config.date_end,
        ],
    )
    assert "--min-listing-trading-days" in calls[1][1]
    assert "--windows" in calls[2][1]
    assert "--horizons" in calls[2][1]
```

Import wrappers at the top of the test file:

```python
from quant_research_platform.pipelines.a_share.full_build_runner import (
    _run_build_factors_step,
    _run_build_interim_step,
    _run_build_panel_step,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_full_build_runner.py::test_full_build_step_wrappers_pass_expected_arguments
```

Expected: FAIL because wrapper stubs raise `NotImplementedError`.

- [ ] **Step 3: Import module `main()` functions**

At the top of `full_build_runner.py`:

```python
from quant_research_platform.markets.implementations.a_share.csmar.build_interim import (
    main as build_interim_main,
)
from quant_research_platform.markets.implementations.a_share.datasets.build_panel import (
    main as build_panel_main,
)
from quant_research_platform.markets.implementations.a_share.factors.build_factors import (
    main as build_factors_main,
)
```

- [ ] **Step 4: Implement wrapper functions**

Replace the stubs:

```python
def _run_build_interim_step(config: PipelineConfig) -> dict[str, Any]:
    argv = [
        "--raw-root",
        config.raw_root.as_posix(),
        "--output-root",
        config.interim_root.as_posix(),
        "--start-date",
        config.date_start,
        "--end-date",
        config.date_end,
    ]
    _require_zero_exit("build_interim", build_interim_main(argv))
    return {"artifact": "interim_private/csmar", "row_count": None}


def _run_build_panel_step(config: PipelineConfig) -> dict[str, Any]:
    argv = [
        "--interim-root",
        config.interim_root.as_posix(),
        "--processed-root",
        config.processed_root.as_posix(),
        "--start-year",
        str(_start_year(config)),
        "--end-year",
        str(_end_year(config)),
        "--end-date",
        config.date_end,
        "--min-listing-trading-days",
        str(config.min_listing_trading_days),
    ]
    _require_zero_exit("build_panel", build_panel_main(argv))
    return {"artifact": "processed_private/csmar/panel_full", "row_count": None}


def _run_build_factors_step(config: PipelineConfig) -> dict[str, Any]:
    argv = [
        "--processed-root",
        config.processed_root.as_posix(),
        "--start-year",
        str(_start_year(config)),
        "--end-year",
        str(_end_year(config)),
        "--factor-set",
        "price_volume",
        "--windows",
        ",".join(str(value) for value in config.factor_windows),
        "--horizons",
        ",".join(str(value) for value in config.label_horizons),
    ]
    _require_zero_exit("build_factors", build_factors_main(argv))
    return {
        "artifact": "processed_private/csmar/price_volume_factors_full",
        "row_count": None,
    }
```

Add helpers:

```python
def _require_zero_exit(step: str, code: int | None) -> None:
    if code not in (0, None):
        raise RuntimeError(f"{step} returned nonzero exit code")


def _start_year(config: PipelineConfig) -> int:
    return int(config.date_start[:4])


def _end_year(config: PipelineConfig) -> int:
    return int(config.date_end[:4])
```

- [ ] **Step 5: Run wrapper test**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_full_build_runner.py::test_full_build_step_wrappers_pass_expected_arguments
```

Expected: PASS.

## Task 6: Wire Evaluation Step

**Files:**
- Modify: `src/quant_research_platform/pipelines/a_share/full_build_runner.py`
- Test: `tests/pipelines/a_share/test_full_build_runner.py`

- [ ] **Step 1: Write evaluation wrapper test**

Append:

```python
def test_full_build_evaluate_step_calls_processed_adapter(tmp_path, monkeypatch) -> None:
    config = _full_config(tmp_path)
    run_dir = tmp_path / "runs" / "full_run"
    run_dir.mkdir(parents=True)
    calls = []

    def fake_run_processed_factor_evaluation(**kwargs):
        calls.append(kwargs)
        report_root = kwargs["report_root"]
        report_root.mkdir(parents=True)
        manifest_path = report_root / "evaluation_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return (
            __import__("pandas").DataFrame([{"label": "forward_return_1d", "rows": 7}]),
            manifest_path,
            [],
        )

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "run_processed_factor_evaluation",
        fake_run_processed_factor_evaluation,
    )

    payload = _run_evaluate_step(config, run_dir)

    assert payload["row_count"] == 7
    assert payload["artifact"] == "evaluation/evaluation_manifest.json"
    assert payload["evaluation_manifest_path"] == run_dir / "evaluation/evaluation_manifest.json"
    assert calls[0]["processed_root"] == config.processed_root
    assert calls[0]["report_root"] == run_dir / "evaluation"
    assert calls[0]["start_year"] == 2024
    assert calls[0]["end_year"] == 2024
    assert calls[0]["requested_labels"] == ("forward_return_1d",)
    assert calls[0]["source_alias"] == "pipeline"
    assert calls[0]["manifest_run_id"] == "full_run"
```

Import `_run_evaluate_step` with the other wrappers.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_full_build_runner.py::test_full_build_evaluate_step_calls_processed_adapter
```

Expected: FAIL because `_run_evaluate_step()` is still a stub.

- [ ] **Step 3: Import and implement evaluation adapter call**

Add import:

```python
from quant_research_platform.markets.implementations.a_share.evaluation.evaluation_adapter import (
    run_processed_factor_evaluation,
)
from quant_research_platform.pipelines.a_share.runner import (
    resolve_processed_factor_columns,
)
```

Replace `_run_evaluate_step()`:

```python
def _run_evaluate_step(config: PipelineConfig, run_dir: Path) -> dict[str, Any]:
    label_summary, manifest_path, _evaluation_runs = run_processed_factor_evaluation(
        processed_root=config.processed_root,
        report_root=run_dir / "evaluation",
        start_year=_start_year(config),
        end_year=_end_year(config),
        requested_labels=config.label_columns,
        requested_factors=resolve_processed_factor_columns(config),
        all_factors=False,
        n_groups=5,
        start_date=config.date_start,
        end_date=config.date_end,
        pnl_return_column=config.pnl_return_column,
        cost_bps=config.cost_bps,
        portfolio_mode=config.portfolio_mode,
        engine=config.engine,
        source_alias="pipeline",
        manifest_run_id=run_dir.name,
    )
    row_count = int(label_summary["rows"].sum()) if "rows" in label_summary else None
    return {
        "artifact": "evaluation/evaluation_manifest.json",
        "row_count": row_count,
        "evaluation_manifest_path": manifest_path,
    }
```

- [ ] **Step 4: Run evaluation wrapper test**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_full_build_runner.py::test_full_build_evaluate_step_calls_processed_adapter
```

Expected: PASS.

## Task 7: Persist Full Build Runtime Summary And Failure State

**Files:**
- Modify: `src/quant_research_platform/pipelines/a_share/full_build_runner.py`
- Test: `tests/pipelines/a_share/test_full_build_runner.py`

- [ ] **Step 1: Write success summary test**

Append:

```python
def test_full_build_runner_writes_runtime_summary_and_checkpoints(tmp_path, monkeypatch) -> None:
    config = _full_config(tmp_path)
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_build_interim_step",
        lambda config: {"row_count": 1, "artifact": "interim"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_build_panel_step",
        lambda config: {"row_count": 2, "artifact": "panel"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_build_factors_step",
        lambda config: {"row_count": 3, "artifact": "factors"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_evaluate_step",
        lambda config, run_dir: {
            "row_count": 4,
            "artifact": "evaluation/evaluation_manifest.json",
            "evaluation_manifest_path": run_dir / "evaluation/evaluation_manifest.json",
        },
    )

    result = run_full_build_pipeline(config)

    summary = json.loads(result.runtime_summary_path.read_text(encoding="utf-8"))
    checkpoints = json.loads((result.run_dir / "checkpoints.json").read_text(encoding="utf-8"))
    assert summary["execution_mode"] == "full_build"
    assert summary["status"] == "ready"
    assert summary["failed_step"] is None
    assert [checkpoint["status"] for checkpoint in checkpoints] == [
        "ready",
        "ready",
        "ready",
        "ready",
        "ready",
    ]
```

Add `import json` if not already present.

- [ ] **Step 2: Write failure sanitization test**

Append:

```python
def test_full_build_runner_records_failed_step_without_private_error_text(tmp_path, monkeypatch) -> None:
    config = _full_config(tmp_path)
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_build_interim_step",
        lambda config: {"row_count": 1, "artifact": "interim"},
    )

    def fail_panel(config):
        raise FileNotFoundError("missing E:/private/raw/vendor/path.csv")

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.full_build_runner."
        "_run_build_panel_step",
        fail_panel,
    )

    with pytest.raises(PipelineRunnerError, match="full pipeline failed at build_panel"):
        run_full_build_pipeline(config)

    summary = json.loads(
        (config.output_root / "full_run" / "runtime_summary.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoints = json.loads(
        (config.output_root / "full_run" / "checkpoints.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "failed"
    assert summary["failed_step"] == "build_panel"
    assert checkpoints[2]["status"] == "failed"
    assert "E:/private" not in json.dumps(summary)
    assert "vendor/path" not in json.dumps(checkpoints)
```

Add imports:

```python
import pytest

from quant_research_platform.pipelines.a_share.runner import PipelineRunnerError
```

- [ ] **Step 3: Run tests to verify behavior**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_full_build_runner.py
```

Expected: PASS after the summary helpers are implemented.

- [ ] **Step 4: Implement `_write_runtime_summary()`**

Use this shape:

```python
def _write_runtime_summary(
    run_dir: Path,
    *,
    config: PipelineConfig,
    status: str,
    checkpoints: list[PipelineCheckpoint],
    warnings: list[dict[str, object]],
    failed_step: str | None,
    failure_code: str | None,
    started: float,
) -> Path:
    payload = {
        "run_id": run_dir.name,
        "requested_run_id": config.run_id,
        "status": status,
        "execution_mode": "full_build",
        "date_start": config.date_start,
        "date_end": config.date_end,
        "engine": config.engine,
        "duration_seconds": _elapsed(started),
        "warnings": warnings,
        "failed_step": failed_step,
        "failure_code": failure_code,
        "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
    }
    path = run_dir / "runtime_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
```

Use the same `asdict(checkpoint)` shape for `_write_checkpoints()`.

## Task 8: Expose Full Build Through Pipeline API Service

**Files:**
- Modify: `src/quant_research_platform/interfaces/api/services/pipeline_service.py`
- Test: create or modify the narrowest existing API service test. If no service test exists, add `tests/interfaces/api/test_pipeline_service.py`.

- [ ] **Step 1: Locate existing API pipeline tests**

Run:

```powershell
rg -n "create_pipeline_snapshot_payload|execution_mode|pipeline-runs" tests
```

Expected: identify whether route/service tests already exist.

- [ ] **Step 2: Write failing service test**

If no suitable test exists, create `tests/interfaces/api/test_pipeline_service.py`:

```python
from quant_research_platform.interfaces.api.services.pipeline_service import (
    PipelineRunRequestPayload,
    create_pipeline_snapshot_payload,
)


def test_pipeline_service_runs_full_build_mode(tmp_path, monkeypatch) -> None:
    calls = []

    class FakeResult:
        status = "ready"
        run_dir = tmp_path / "runs" / "full_run"
        warnings = []

    def fake_run_full_build_pipeline(config):
        calls.append(config)
        return FakeResult()

    monkeypatch.setattr(
        "quant_research_platform.interfaces.api.services.pipeline_service."
        "run_full_build_pipeline",
        fake_run_full_build_pipeline,
    )
    payload = PipelineRunRequestPayload(
        run_id="full_run",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="rev_20d",
        factor_template_id="custom",
        label_columns=["forward_return_1d"],
        execution_mode="full_build",
    )
    state = {}

    result = create_pipeline_snapshot_payload(
        payload,
        state,
        output_root=tmp_path / "runs",
    )

    assert result["status"] == "ready"
    assert result["run_ref"] == "full_run"
    assert calls[0].processed_root.as_posix().endswith("data/processed_private/csmar")
    assert state["full_run"].run_ref == "full_run"
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/interfaces/api/test_pipeline_service.py::test_pipeline_service_runs_full_build_mode
```

Expected: FAIL because `full_build` is unsupported or `run_full_build_pipeline` is not imported.

- [ ] **Step 4: Import and call full runner**

Modify imports:

```python
from quant_research_platform.pipelines.a_share.full_build_runner import (
    run_full_build_pipeline,
)
```

Add branch in `create_pipeline_snapshot_payload()` after `processed_existing`:

```python
    elif payload.execution_mode == "full_build":
        try:
            result = run_full_build_pipeline(config)
        except PipelineRunnerError as exc:
            run_dir = getattr(exc, "run_dir", None)
            if run_dir is None:
                raise HTTPException(
                    status_code=422,
                    detail="pipeline execution failed",
                ) from exc
            summary = PipelineRunSummaryPayload(
                run_id=config.run_id,
                status="failed",
                message=_pipeline_failure_message(exc),
                run_ref=run_dir.name,
                warnings=[],
            )
            _remember_pipeline_summary(pipeline_runs, payload.run_id, summary)
            return summary.model_dump()
        run_ref = result.run_dir.name
        summary = PipelineRunSummaryPayload(
            run_id=config.run_id,
            status=result.status,
            message="full-build pipeline 已完成",
            run_ref=run_ref,
            warnings=result.warnings,
        )
```

- [ ] **Step 5: Run API service test**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/interfaces/api/test_pipeline_service.py
```

Expected: PASS.

## Task 9: Keep Existing Processed-Existing Behavior Stable

**Files:**
- Modify only if tests expose a regression:
  - `src/quant_research_platform/pipelines/a_share/runner.py`
  - `src/quant_research_platform/interfaces/api/services/pipeline_service.py`
- Test: `tests/pipelines/a_share/test_pipeline_runner.py`

- [ ] **Step 1: Run existing processed-existing tests**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share/test_pipeline_runner.py
```

Expected: PASS.

- [ ] **Step 2: Fix only direct regressions**

If this fails, inspect the failing assertion and fix only the behavior changed by N1. Do not rename `processed_evaluation`, do not change `execution_mode == "processed_existing"` summaries, and do not alter existing `source_alias="pipeline"` behavior unless a test proves the current behavior is wrong.

## Task 10: Documentation Updates

**Files:**
- Modify: `docs/architecture/next-stage-roadmap.md`
- Modify: `docs/code-map/platform-flow.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Update roadmap after code passes**

In `docs/architecture/next-stage-roadmap.md`, under N1, add a short implementation note:

```markdown
Implementation note:

- `full_build` mode now orchestrates interim, panel, factor, and evaluation steps through `pipelines/a_share/full_build_runner.py`.
- `processed_existing` remains the lightweight runner for existing processed factor tables.
```

Only add this after the code and tests pass.

- [ ] **Step 2: Update code map**

In `docs/code-map/platform-flow.md`, update the pipeline section so it lists both:

```text
processed_existing -> existing processed factor table -> evaluation
full_build -> build_interim -> build_panel -> build_factors -> evaluation
```

- [ ] **Step 3: Update project status**

In `PROJECT_STATUS.md`, update the current status and root handoff only if this branch changes the current project state. If only the plan is written or code is not merged, do not claim N1 complete.

## Task 11: Verification

**Files:**
- No edits unless verification exposes a bug.

- [ ] **Step 1: Run focused backend tests**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share tests/markets/implementations/a_share
```

Expected: PASS.

- [ ] **Step 2: Run API tests touched by service changes**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/interfaces/api
```

Expected: PASS.

- [ ] **Step 3: Run lint**

Run:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m ruff check src tests
```

Expected: PASS.

- [ ] **Step 4: Run diff check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

## Self-Review Checklist

- [ ] The plan preserves `processed_existing`.
- [ ] The plan does not expose private CSMAR paths or raw exception text in runtime artifacts.
- [ ] The plan does not add external data, credentials, deployment, paper trading, or live trading.
- [ ] Every new user-visible artifact is created through pipeline output or evaluation manifest logic.
- [ ] Every code step has a narrow test and an expected failure/pass command.
