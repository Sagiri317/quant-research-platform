# N2 Tradability Artifact/API Implementation Plan

> Status: completed backend artifact/API layer on 2026-05-24. This file is retained as the implementation record; current contract truth is the source code plus `docs/code-map/evaluation-artifacts-api.md`.
>
> **For agentic workers:** Historical execution plan. Do not treat unchecked checklist items below as current pending work unless a later task explicitly reopens them.

**Goal:** Add a backend-only N2 tradability diagnostics contract so eligibility, entry blocking, and exit blocking are exposed through schema `0.5` artifacts and evidence API routes without changing frontend display code.

**Architecture:** Keep generated evaluation artifacts as the source of truth. Add a dedicated `diagnostics/tradability_diagnostics.csv` table to the evaluation-set artifact contract, include it in `manifest.display`, and let the existing `/diagnostics/{metric_name}` and direct download routes expose it through the same display gate as other diagnostics tables. Do not add frontend-specific shortcuts or duplicate calculations in API routes.

**Tech Stack:** Python, pandas, FastAPI, pytest, Conda `analysis`, PowerShell on Windows.

---

## Scope

Implement N2 backend artifact/API only:

- Add one schema `0.5` diagnostics table: `tradability_diagnostics`.
- Derive it from existing `weight_diagnostics` rows emitted by dataframe and tensor portfolio paths.
- Expose it through `manifest.display.tables`, `manifest.display.downloads.items`, download presets, CSV preview, direct CSV download, ZIP download, and evidence artifact loading.
- Keep `source_alias + run_id` identity unchanged.
- Keep the existing `weight_diagnostics` table intact for backward-compatible internal diagnostics inside the current schema version.

Do not:

- Modify `web/`.
- Add React types, components, charts, tabs, copy, CSS, browser tests, or frontend state changes.
- Regenerate private CSMAR data or private real reports as part of this plan.
- Change artifact schema version from `0.5`.
- Add dependencies.
- Introduce API routes that bypass `manifest.display`.

## Initial Diagnosis Before Implementation

Backend state before this plan was implemented:

- `weight_diagnostics` already contains tradability counts: `universe_count`, `eligible_count`, `can_enter_count`, `can_exit_count`, `entry_blocked_count`, and `exit_blocked_count`.
- `EVALUATION_SET_DIAGNOSTIC_TABLES` included only `weight_diagnostics` and `neutralization_diagnostics`.
- `/api/runs/{source_alias}/{run_id}/diagnostics/{metric_name}` already previews any diagnostics CSV that is listed in `manifest.display.tables`.
- Direct and ZIP downloads already validate requested items through `manifest.display.downloads.items`.
- `/evidence` eager-loads only `combinations` and chart-backed tables, so a backend-only N2 artifact does not need to change evidence payload behavior unless a later frontend task wants first-load rendering.

The missing backend contract is a dedicated display-exposed table whose name and columns make tradability diagnostics explicit instead of requiring users to know that the data is hidden inside `weight_diagnostics`.

## Public Artifact Contract

Add this table to evaluation-set schema `0.5`:

```text
diagnostics/tradability_diagnostics.csv
```

This table is required for newly generated schema `0.5` evaluation-set runs.
Legacy runs that do not declare and write this table are intentionally
incompatible with the current loader/API and must be regenerated.

Table key:

```text
tradability_diagnostics
```

Required columns:

```text
combination_id
feature
label
weight_id
cost_id
timestamp
method
direction
universe_count
eligible_count
can_enter_count
can_exit_count
entry_blocked_count
exit_blocked_count
eligible_rate
can_enter_rate
can_exit_rate
entry_blocked_rate
exit_blocked_rate
fallback_reason
```

Column semantics:

- `universe_count`: rows in the timestamp/feature universe after panel validation.
- `eligible_count`: rows with non-null signal and passing `eligibility_column` when configured.
- `can_enter_count`: eligible rows that pass `can_enter_column` when configured.
- `can_exit_count`: rows that pass `can_exit_column` when configured.
- `entry_blocked_count`: `eligible_count - can_enter_count`, floored at zero.
- `exit_blocked_count`: `universe_count - can_exit_count`, floored at zero.
- `eligible_rate`: `eligible_count / universe_count`, or `0.0` when denominator is zero.
- `can_enter_rate`: `can_enter_count / eligible_count`, or `0.0` when denominator is zero.
- `can_exit_rate`: `can_exit_count / universe_count`, or `0.0` when denominator is zero.
- `entry_blocked_rate`: `entry_blocked_count / eligible_count`, or `0.0` when denominator is zero.
- `exit_blocked_rate`: `exit_blocked_count / universe_count`, or `0.0` when denominator is zero.
- `fallback_reason`: copied from `weight_diagnostics` so blocked-tradability effects and weight fallback effects can be inspected together.

Do not include raw private paths, CSMAR file paths, local drive letters, or asset-level holdings details in this table.

## File Responsibilities

- `src/quant_research_platform/artifacts/artifact_schema.py`: add `tradability_diagnostics` to diagnostics schema and path contract.
- `src/quant_research_platform/artifacts/display_manifest.py`: add label and group metadata so the new table appears in `manifest.display`.
- `src/quant_research_platform/artifacts/manifest_validation.py`: require dimension columns and the concrete N2 tradability diagnostic columns for `tradability_diagnostics`.
- `src/quant_research_platform/evaluations/evaluation_runner.py`: build `tradability_diagnostics` from `weight_diagnostics` in both dataframe and tensor paths.
- `tests/evaluations/test_evaluation_engine_contract.py`: verify the new table exists and contains rates.
- `tests/artifacts/artifact_fixtures.py`: update fixture artifacts and manifests.
- `tests/interfaces/api/artifact_run_fixtures.py`: update API fixture artifacts and manifests.
- `tests/artifacts/test_artifact_contract.py`: verify artifact loading/display contract includes the new diagnostics table.
- `tests/interfaces/api/test_api.py`: verify preview and direct download for `diagnostics/tradability_diagnostics.csv`.
- `docs/code-map/evaluation-artifacts-api.md`: after implementation only, document the new backend artifact/API contract. Keep frontend wording out of this backend plan unless frontend work is separately approved.

## Task 1: Add Artifact Schema Contract

**Files:**

- Modify: `src/quant_research_platform/artifacts/artifact_schema.py`
- Modify: `src/quant_research_platform/artifacts/display_manifest.py`
- Modify: `src/quant_research_platform/artifacts/manifest_validation.py`
- Test: `tests/artifacts/test_artifact_contract.py`

- [ ] **Step 1: Write failing artifact contract assertion**

Add an assertion in `tests/artifacts/test_artifact_contract.py` near the existing required-table/display checks:

```python
assert "tradability_diagnostics" in loaded.tables
display_tables = {
    table["key"]: table
    for table in loaded.manifest["display"]["tables"]
}
assert display_tables["tradability_diagnostics"]["path"] == (
    "diagnostics/tradability_diagnostics.csv"
)
download_items = {
    item["item"]
    for item in loaded.manifest["display"]["downloads"]["items"]
}
assert "diagnostics/tradability_diagnostics.csv" in download_items
```

- [ ] **Step 2: Run the failing artifact test**

Run:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src'
conda run -n analysis python -m pytest -q tests\artifacts\test_artifact_contract.py
```

Expected: fail because `tradability_diagnostics` is not yet part of the artifact fixture/schema.

- [ ] **Step 3: Add schema path**

In `src/quant_research_platform/artifacts/artifact_schema.py`, update diagnostics constants and path contract:

```python
EVALUATION_SET_DIAGNOSTIC_TABLES = (
    "weight_diagnostics",
    "neutralization_diagnostics",
    "tradability_diagnostics",
)
```

Inside the `"diagnostics"` artifact section, add:

```python
"tradability_diagnostics": _relative(
    base,
    base / "diagnostics" / "tradability_diagnostics.csv",
),
```

- [ ] **Step 4: Add display label**

In `src/quant_research_platform/artifacts/display_manifest.py`, add:

```python
"tradability_diagnostics": ("可交易性诊断", "diagnostics"),
```

Keep it next to `weight_diagnostics` and `neutralization_diagnostics`.

- [ ] **Step 5: Add column validation group**

In `src/quant_research_platform/artifacts/manifest_validation.py`, validate `tradability_diagnostics` with its concrete N2 contract columns, not only the generic dimension columns:

```python
TRADABILITY_DIAGNOSTIC_COLUMNS = frozenset(
    {
        "timestamp",
        "method",
        "direction",
        "universe_count",
        "eligible_count",
        "can_enter_count",
        "can_exit_count",
        "entry_blocked_count",
        "exit_blocked_count",
        "eligible_rate",
        "can_enter_rate",
        "can_exit_rate",
        "entry_blocked_rate",
        "exit_blocked_rate",
        "fallback_reason",
    }
)
```

- [ ] **Step 6: Run artifact tests**

Run:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src'
conda run -n analysis python -m pytest -q tests\artifacts\test_artifact_contract.py tests\artifacts\test_artifact_package.py
```

Expected: failures remain only where fixtures or producer output do not yet include the new table.

## Task 2: Produce Tradability Diagnostics Table

**Files:**

- Modify: `src/quant_research_platform/evaluations/evaluation_runner.py`
- Test: `tests/evaluations/test_evaluation_engine_contract.py`

- [ ] **Step 1: Write failing evaluation assertion**

In `tests/evaluations/test_evaluation_engine_contract.py`, extend the existing evaluation contract test that checks `weight_diagnostics`:

```python
assert "tradability_diagnostics" in result.metrics
tradability = result.metrics["tradability_diagnostics"]
assert {
    "eligible_rate",
    "can_enter_rate",
    "can_exit_rate",
    "entry_blocked_rate",
    "exit_blocked_rate",
}.issubset(tradability.columns)
assert (
    tradability[
        [
            "eligible_rate",
            "can_enter_rate",
            "can_exit_rate",
            "entry_blocked_rate",
            "exit_blocked_rate",
        ]
    ]
    .apply(pd.to_numeric, errors="coerce")
    .ge(0.0)
    .all()
    .all()
)
```

- [ ] **Step 2: Run the failing evaluation test**

Run:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src'
conda run -n analysis python -m pytest -q tests\evaluations\test_evaluation_engine_contract.py
```

Expected: fail because `tradability_diagnostics` is not produced yet.

- [ ] **Step 3: Add helper to derive rates**

In `src/quant_research_platform/evaluations/evaluation_runner.py`, add a small private helper near `_metrics_for_feature`:

```python
def _build_tradability_diagnostics(weight_diagnostics: pd.DataFrame) -> pd.DataFrame:
    result = weight_diagnostics.copy()
    if result.empty:
        return result.assign(
            eligible_rate=pd.Series(dtype="float64"),
            can_enter_rate=pd.Series(dtype="float64"),
            can_exit_rate=pd.Series(dtype="float64"),
            entry_blocked_rate=pd.Series(dtype="float64"),
            exit_blocked_rate=pd.Series(dtype="float64"),
        )
    result["eligible_rate"] = _safe_rate(
        result["eligible_count"],
        result["universe_count"],
    )
    result["can_enter_rate"] = _safe_rate(
        result["can_enter_count"],
        result["eligible_count"],
    )
    result["can_exit_rate"] = _safe_rate(
        result["can_exit_count"],
        result["universe_count"],
    )
    result["entry_blocked_rate"] = _safe_rate(
        result["entry_blocked_count"],
        result["eligible_count"],
    )
    result["exit_blocked_rate"] = _safe_rate(
        result["exit_blocked_count"],
        result["universe_count"],
    )
    return result


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    bases = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return values.div(bases.where(bases.ne(0.0), other=pd.NA)).fillna(0.0)
```

- [ ] **Step 4: Include the table in both portfolio paths**

In both `_run_dataframe_feature_set()` and `_run_tensor_feature_set()`, build the new table from the existing diagnostics frame:

```python
tradability_diagnostics = _build_tradability_diagnostics(diagnostics)
```

For tensor path use:

```python
tradability_diagnostics = _build_tradability_diagnostics(
    portfolio_artifacts.diagnostics
)
```

Add it to each returned `metrics` dictionary:

```python
"tradability_diagnostics": tradability_diagnostics,
```

- [ ] **Step 5: Initialize aggregation bucket**

In `evaluate_to_payload()`, add the table to `all_tables`:

```python
"tradability_diagnostics": [],
```

- [ ] **Step 6: Run evaluation tests**

Run:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src'
conda run -n analysis python -m pytest -q tests\evaluations\test_evaluation_engine_contract.py
```

Expected: pass.

## Task 3: Update Artifact And API Fixtures

**Files:**

- Modify: `tests/artifacts/artifact_fixtures.py`
- Modify: `tests/interfaces/api/artifact_run_fixtures.py`
- Test: `tests/artifacts/test_artifact_contract.py`
- Test: `tests/interfaces/api/test_api.py`

- [ ] **Step 1: Add fixture frame helper**

In both fixture files, add a helper matching the fixture style:

```python
def _tradability_diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2024-01-02", "2024-01-03"],
            "feature": ["factor", "factor"],
            "method": ["equal_weight_quantile", "equal_weight_quantile"],
            "direction": ["long_high", "long_high"],
            "universe_count": [100, 100],
            "eligible_count": [90, 80],
            "can_enter_count": [85, 70],
            "can_exit_count": [95, 90],
            "entry_blocked_count": [5, 10],
            "exit_blocked_count": [5, 10],
            "long_count": [20, 18],
            "short_count": [20, 18],
            "eligible_rate": [0.9, 0.8],
            "can_enter_rate": [85 / 90, 70 / 80],
            "can_exit_rate": [0.95, 0.9],
            "entry_blocked_rate": [5 / 90, 10 / 80],
            "exit_blocked_rate": [0.05, 0.1],
            "fallback_reason": ["", ""],
        }
    )
```

- [ ] **Step 2: Write fixture CSV**

Next to `weight_diagnostics.csv` and `neutralization_diagnostics.csv`, write:

```python
_with_dims(_tradability_diagnostics_frame()).to_csv(
    output_dir / "diagnostics" / "tradability_diagnostics.csv",
    index=False,
)
```

- [ ] **Step 3: Add manifest path**

In fixture manifest `artifacts["diagnostics"]`, add:

```python
"tradability_diagnostics": "diagnostics/tradability_diagnostics.csv",
```

- [ ] **Step 4: Run artifact/API fixture-backed tests**

Run:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src'
conda run -n analysis python -m pytest -q tests\artifacts\test_artifact_contract.py tests\interfaces\api\test_api.py
```

Expected: any remaining failures should point to missing API assertions, not missing fixture files.

## Task 4: Add API Preview And Download Tests

**Files:**

- Modify: `tests/interfaces/api/test_api.py`
- Test: `tests/interfaces/api/test_api.py`

- [ ] **Step 1: Add diagnostics preview test**

Add:

```python
def test_api_previews_tradability_diagnostics_table(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/diagnostics/tradability_diagnostics"
    )

    assert response.status_code == 200
    payload = response.json()
    assert "eligible_rate" in payload["columns"]
    assert "entry_blocked_rate" in payload["columns"]
```

- [ ] **Step 2: Add direct download test**

Add:

```python
def test_api_downloads_tradability_diagnostics_artifact_csv(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/artifacts/diagnostics/tradability_diagnostics.csv"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="tradability_diagnostics.csv"' in response.headers[
        "content-disposition"
    ]
    assert "eligible_rate" in response.text
    assert "entry_blocked_rate" in response.text
```

- [ ] **Step 3: Add ZIP selected item assertion**

Extend the selected ZIP test that already checks diagnostics/download item behavior with:

```python
"diagnostics/tradability_diagnostics.csv",
```

When inspecting archive contents, assert:

```python
assert b"eligible_rate" in archive.read(
    "diagnostics/tradability_diagnostics.csv"
)
```

- [ ] **Step 4: Run API tests**

Run:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src'
conda run -n analysis python -m pytest -q tests\interfaces\api\test_api.py
```

Expected: pass.

## Task 5: Documentation And Code Map Sync

**Files:**

- Modify: `docs/code-map/evaluation-artifacts-api.md`
- Modify if status changes: `PROJECT_STATUS.md`

- [ ] **Step 1: Update code map artifact list**

In `docs/code-map/evaluation-artifacts-api.md`, add:

```text
diagnostics/tradability_diagnostics.csv
```

Keep it near the existing diagnostics files.

- [ ] **Step 2: Update manifest.display description**

Add one sentence:

```text
N2 tradability diagnostics are exposed as `diagnostics/tradability_diagnostics.csv`
through `manifest.display`; frontend code must consume this table through the
normal diagnostics preview/download APIs and must not recalculate these counts.
```

- [x] **Step 3: Update project status only if N2 backend is complete**

If all source and tests are complete, add an `open` or `closed` root handoff row according to the project `AGENTS.md` root handoff rule. Use concise wording:

```text
N2 backend artifact/API contract added for tradability diagnostics; frontend
visualization remains a separate task.
```

- [x] **Step 4: Run docs-safe checks**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

## Final Verification

Run the narrow backend verification set:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src'
conda run -n analysis python -m pytest -q tests\evaluations tests\artifacts tests\interfaces\api
conda run -n analysis python -m ruff check src tests
git diff --check
```

Expected:

- All selected backend tests pass.
- Ruff passes.
- `git diff --check` passes.

Windows note: `conda run` may print OpenCL vendor noise even when tests pass. Judge success by process exit code and pytest/ruff summaries.

## Review Checklist

- `tradability_diagnostics` exists in artifact schema, writer output, loader output, fixture manifests, and `manifest.display`.
- API preview uses `/api/runs/{source_alias}/{run_id}/diagnostics/tradability_diagnostics`.
- Direct and ZIP downloads are gated by `manifest.display.downloads.items`.
- No frontend file under `web/` changed.
- No private data path appears in new artifact payloads, docs, errors, or tests.
- `source_alias + run_id` identity remains unchanged.
- `weight_diagnostics` remains available.
