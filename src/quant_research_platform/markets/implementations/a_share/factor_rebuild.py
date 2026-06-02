"""A-share literature-prior factor rebuild service."""

from __future__ import annotations

import json
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.markets.implementations.a_share.csmar.columns import (  # noqa: E501
    SLIM_FACTOR_BASE_COLUMNS,
)
from quant_research_platform.markets.implementations.a_share.factors.labels import (
    add_forward_return_labels,
)
from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
    LITERATURE_FACTOR_SET_TIERS,
    LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
    LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID,
    calculate_literature_price_volume_factors,
    expected_literature_price_volume_factor_columns,
    largest_literature_price_volume_lookback_days,
    select_literature_price_volume_features,
    unsupported_literature_price_volume_factors,
)
from quant_research_platform.markets.implementations.a_share.research_batch_safety import (  # noqa: E501
    build_schema_inventory,
    run_memory_preflight,
    write_json_sanitized,
)
from quant_research_platform.runtime.private_paths import (
    path_has_private_marker,
    path_is_under_private_output_root,
    require_private_outputs_for_private_input,
    text_has_private_marker,
)

_UNSAFE_RUN_ID_MARKER_PATTERN = re.compile(
    r"(^|[_-])(vendor|source|local)([_-]|$)",
    re.IGNORECASE,
)
_SENSITIVE_RUN_ID_PATTERN = re.compile(
    r"(credential|secret|token|api[_-]?key|password)",
    re.IGNORECASE,
)
_UNSAFE_RUN_ID_MARKERS = frozenset({"vendor", "source", "local"})

PREDICTION_SAFE_BOUNDARY = (
    "experimental research factor rebuild; not alpha proof; not investment advice; "
    "not trading, paper trading, live trading, risk, capacity, or execution evidence"
)

FACTOR_REBUILD_INPUT_COLUMNS: tuple[str, ...] = (
    "stock_code",
    "trade_date",
    "market_type",
    "eligible",
    "can_enter",
    "can_exit",
    "is_a_share_scope",
    "adj_open",
    "adj_close",
    "return_1d",
    "amount",
    "turnover",
    "high",
    "low",
    "prev_close",
    "float_market_cap",
    "total_market_cap",
)


@dataclass(frozen=True)
class FactorRebuildConfig:
    """Config for one local A-share factor rebuild run."""

    panel_parquet: Path
    output_root: Path
    run_id: str
    factor_pack: str = LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID
    memory_budget_gb: float = 10.0
    dry_run: bool = False
    factor_parquet: Path | None = None
    overwrite_processed: bool = False
    backup_existing: bool = False
    start_date: str | None = None
    end_date: str | None = None
    calculation_start_date: str | None = None
    evaluation_start_date: str | None = None
    factor_set_tier: str = "full"
    require_sufficient_warmup: bool = False


@dataclass(frozen=True)
class FactorRebuildResult:
    """Small summary of one factor rebuild run."""

    run_dir: Path
    status: str
    row_count: int
    output_parquet: Path | None = None
    overwritten_parquet: Path | None = None


def write_schema_inventory_run(
    *,
    input_parquet: Path,
    output_root: Path,
    run_id: str,
    memory_budget_gb: float = 10.0,
) -> Path:
    """Write aggregate-only schema inventory and memory preflight artifacts."""

    require_private_outputs_for_private_input(
        input_parquet,
        (output_root,),
        error_message="private_output_policy_violation",
    )
    run_dir = _new_run_dir(output_root, run_id)
    preflight = run_memory_preflight(
        input_parquet,
        required_columns=("stock_code", "trade_date"),
        memory_budget_gb=memory_budget_gb,
    )
    write_json_sanitized(run_dir / "memory_preflight.json", preflight)
    if preflight["status"] != "passed":
        _write_failed_run_summary(run_dir, run_id, "memory_preflight_failed")
        _write_checkpoints(run_dir)
        raise ValueError("memory_preflight_failed")
    inventory = build_schema_inventory(
        input_parquet,
        expected_columns=("stock_code", "trade_date"),
        start_date=None,
        end_date=None,
    )
    write_json_sanitized(run_dir / "schema_inventory.json", inventory)
    _write_checkpoints(run_dir)
    write_json_sanitized(
        run_dir / "runtime_summary.json",
        {
            "status": "ready",
            "run_id": run_id,
            "artifact_type": "schema_inventory",
            "boundary": PREDICTION_SAFE_BOUNDARY,
        },
    )
    return run_dir


def run_factor_rebuild(config: FactorRebuildConfig) -> FactorRebuildResult:
    """Run or dry-run one literature-prior factor rebuild."""

    _validate_run_id_available(config.output_root, config.run_id)
    _validate_rebuild_config(config)
    run_dir = _new_run_dir(config.output_root, config.run_id)
    try:
        available_input_columns = tuple(
            pq.ParquetFile(config.panel_parquet).schema.names
        )
        expected_factors = expected_literature_price_volume_factor_columns(
            factor_set_tier=config.factor_set_tier
        )
        preflight = run_memory_preflight(
            config.panel_parquet,
            required_columns=("stock_code", "trade_date", "adj_open", "adj_close"),
            selected_columns=_available_columns(
                config.panel_parquet,
                FACTOR_REBUILD_INPUT_COLUMNS,
            ),
            estimated_output_columns=(
                len(SLIM_FACTOR_BASE_COLUMNS) + len(expected_factors) + 6
            ),
            date_column="trade_date",
            start_date=_calculation_start_date(config),
            end_date=config.end_date,
            memory_budget_gb=config.memory_budget_gb,
        )
        write_json_sanitized(run_dir / "memory_preflight.json", preflight)
        if preflight["status"] != "passed":
            raise ValueError("memory_preflight_failed")
        inventory = build_schema_inventory(
            config.panel_parquet,
            expected_columns=FACTOR_REBUILD_INPUT_COLUMNS,
            start_date=_calculation_start_date(config),
            end_date=config.end_date,
        )
        write_json_sanitized(run_dir / "schema_inventory.json", inventory)
        warmup = build_warmup_sufficiency_payload(
            config=config,
            expected_factors=expected_factors,
        )
        write_json_sanitized(run_dir / "warmup_sufficiency.json", warmup)
        if config.require_sufficient_warmup and not warmup["warmup_sufficient"]:
            raise ValueError("warmup_insufficient")
        unsupported = unsupported_literature_price_volume_factors(
            available_input_columns,
            expected_columns=expected_factors,
        )
        _write_feature_set_manifest(
            run_dir=run_dir,
            feature_columns=(),
            unsupported_factors=unsupported,
            warnings=(),
            row_count=0,
            factor_set_tier=config.factor_set_tier,
        )
        if config.dry_run:
            write_json_sanitized(
                run_dir / "runtime_summary.json",
                {
                    **_runtime_summary(config.run_id, "dry_run", None),
                    "warmup_status": warmup["warmup_status"],
                    "warmup": warmup,
                },
            )
            _write_checkpoints(run_dir)
            return FactorRebuildResult(run_dir=run_dir, status="dry_run", row_count=0)

        read_columns = _available_columns(
            config.panel_parquet,
            FACTOR_REBUILD_INPUT_COLUMNS,
        )
        panel = pd.read_parquet(
            config.panel_parquet,
            columns=read_columns,
            filters=_date_filters(_calculation_start_date(config), config.end_date),
            engine="pyarrow",
        )
        if "stock_code" in panel.columns:
            panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)
        labeled = add_forward_return_labels(panel)
        factored = calculate_literature_price_volume_factors(
            labeled,
            expected_columns=expected_factors,
        )
        feature_columns, warnings = select_literature_price_volume_features(
            factored,
            expected_columns=expected_factors,
        )
        output = _project_factor_tier_columns(factored, expected_factors)
        output = _crop_to_evaluation_window(output, config=config)
        output = output.sort_values(["stock_code", "trade_date"]).reset_index(
            drop=True
        )
        output_path = run_dir / "factor_rebuild.parquet"
        _write_factor_table(output_path, output)
        run_warnings = _degraded_run_warnings(config, warnings, warmup)
        _write_feature_set_manifest(
            run_dir=run_dir,
            feature_columns=feature_columns,
            unsupported_factors=unsupported,
            warnings=run_warnings,
            row_count=len(output),
            factor_set_tier=config.factor_set_tier,
        )
        _write_factor_eda(
            run_dir=run_dir,
            frame=output,
            feature_columns=feature_columns,
        )
        if config.factor_parquet is not None:
            _replace_processed_factor_table(
                target=config.factor_parquet,
                frame=output,
                run_id=config.run_id,
                backup_existing=config.backup_existing,
                run_dir=run_dir,
            )
        write_json_sanitized(
            run_dir / "runtime_summary.json",
            {
                **_runtime_summary(config.run_id, "ready", None),
                "row_count": int(len(output)),
                "feature_count": len(feature_columns),
                "warning_count": len(run_warnings),
                "unsupported_factor_count": len(unsupported),
                "factor_pack": config.factor_pack,
                "factor_set_tier": config.factor_set_tier,
                "date_filter": {
                    "calculation_start_date": _calculation_start_date(config),
                    "evaluation_start_date": _evaluation_start_date(config),
                    "end_date": config.end_date,
                }
                if _calculation_start_date(config) or config.end_date
                else None,
                "warmup_status": warmup["warmup_status"],
                "warmup": warmup,
            },
        )
        _write_checkpoints(run_dir)
        return FactorRebuildResult(
            run_dir=run_dir,
            status="ready",
            row_count=len(output),
            output_parquet=output_path,
            overwritten_parquet=(
                config.factor_parquet if config.factor_parquet else None
            ),
        )
    except Exception as exc:
        if not (run_dir / "runtime_summary.json").is_file():
            _write_failed_run_summary(run_dir, config.run_id, _failure_code(exc))
        _write_checkpoints(run_dir)
        raise


def _validate_rebuild_config(config: FactorRebuildConfig) -> None:
    if config.factor_pack != LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID:
        raise ValueError("invalid_factor_rebuild_config")
    if config.factor_set_tier not in LITERATURE_FACTOR_SET_TIERS:
        raise ValueError("invalid_factor_rebuild_config")
    calculation_start = _timestamp_or_none(_calculation_start_date(config))
    evaluation_start = _timestamp_or_none(_evaluation_start_date(config))
    if (
        calculation_start is not None
        and evaluation_start is not None
        and calculation_start > evaluation_start
    ):
        raise ValueError("invalid_factor_rebuild_config")
    if not config.panel_parquet.is_file():
        raise FileNotFoundError(config.panel_parquet)
    require_private_outputs_for_private_input(
        config.panel_parquet,
        (config.output_root,),
        error_message="private_output_policy_violation",
    )
    if config.factor_parquet is not None:
        if not path_has_private_marker(config.factor_parquet):
            raise ValueError("private_output_policy_violation")
        if config.factor_parquet.exists() and not config.overwrite_processed:
            raise ValueError("--overwrite-processed is required for replacement")
    if config.backup_existing and not config.overwrite_processed:
        raise ValueError("--backup-existing requires --overwrite-processed")


def _validate_run_id_available(output_root: Path, run_id: str) -> Path:
    if (
        Path(run_id).name != run_id
        or run_id in {"", ".", ".."}
        or _SENSITIVE_RUN_ID_PATTERN.search(run_id)
        or _UNSAFE_RUN_ID_MARKER_PATTERN.search(run_id)
        or _run_id_has_unsafe_marker(run_id)
        or _run_id_has_unsafe_path_text(run_id)
        or text_has_private_marker(run_id)
    ):
        raise ValueError("invalid_factor_rebuild_config")
    run_dir = output_root / run_id
    if run_dir.exists():
        raise ValueError("duplicate_run_id")
    return run_dir


def _new_run_dir(output_root: Path, run_id: str) -> Path:
    run_dir = _validate_run_id_available(output_root, run_id)
    run_dir.mkdir(parents=True)
    return run_dir


def _run_id_has_unsafe_marker(run_id: str) -> bool:
    camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", run_id)
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_split)
    return any(
        token.casefold() in _UNSAFE_RUN_ID_MARKERS
        for token in re.split(r"[^A-Za-z0-9]+", camel_split)
        if token
    )


def _run_id_has_unsafe_path_text(run_id: str) -> bool:
    normalized = run_id.replace("\\", "/")
    if re.search(r"(?<![A-Za-z0-9])[A-Za-z]:/", normalized):
        return True
    if normalized.startswith(("/", "//")):
        return True
    return any(part == ".." for part in normalized.split("/"))


def _available_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    names = pq.ParquetFile(path).schema.names
    return [column for column in wanted if column in names]


def _date_filters(
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, str, object]] | None:
    filters: list[tuple[str, str, object]] = []
    if start_date:
        filters.append(("trade_date", ">=", pd.Timestamp(start_date)))
    if end_date:
        filters.append(("trade_date", "<=", pd.Timestamp(end_date)))
    return filters or None


def _calculation_start_date(config: FactorRebuildConfig) -> str | None:
    return config.calculation_start_date or config.start_date


def _evaluation_start_date(config: FactorRebuildConfig) -> str | None:
    return config.evaluation_start_date or config.start_date


def _timestamp_or_none(value: str | None) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value)


def _crop_to_evaluation_window(
    frame: pd.DataFrame,
    *,
    config: FactorRebuildConfig,
) -> pd.DataFrame:
    start = _evaluation_start_date(config)
    end = config.end_date
    if not (start or end):
        return frame
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= dates >= pd.Timestamp(start)
    if end:
        mask &= dates <= pd.Timestamp(end)
    return frame.loc[mask].copy()


def build_warmup_sufficiency_payload(
    *,
    config: FactorRebuildConfig,
    expected_factors: tuple[str, ...],
) -> dict[str, object]:
    """Build aggregate warmup sufficiency metadata from trading dates only."""

    calculation_start = _calculation_start_date(config)
    evaluation_start = _evaluation_start_date(config)
    largest_lookback = largest_literature_price_volume_lookback_days(
        expected_factors
    )
    observation_count: int | None = None
    if (
        calculation_start
        and evaluation_start
        and pd.Timestamp(calculation_start) < pd.Timestamp(evaluation_start)
    ):
        dates = pd.read_parquet(
            config.panel_parquet,
            columns=("trade_date",),
            filters=_date_filters(calculation_start, None),
            engine="pyarrow",
        )
        values = pd.to_datetime(dates["trade_date"], errors="coerce")
        mask = values < pd.Timestamp(evaluation_start)
        observation_count = int(values.loc[mask].dropna().nunique())
        warmup_sufficient = observation_count >= largest_lookback
        status = "warmup_sufficient" if warmup_sufficient else "warmup_insufficient"
    elif calculation_start or evaluation_start or config.end_date:
        warmup_sufficient = False
        status = "date_filtered_without_warmup"
    else:
        warmup_sufficient = True
        status = "full_range"
    return {
        "warmup_status": status,
        "largest_lookback_days": int(largest_lookback),
        "warmup_observation_count": observation_count,
        "calculation_start_date": calculation_start,
        "evaluation_start_date": evaluation_start,
        "end_date": config.end_date,
        "warmup_sufficient": bool(warmup_sufficient),
    }


def _project_factor_tier_columns(
    frame: pd.DataFrame,
    expected_factors: tuple[str, ...],
) -> pd.DataFrame:
    allowed = set(SLIM_FACTOR_BASE_COLUMNS).union(expected_factors)
    columns = [
        column
        for column in frame.columns
        if column in allowed or column.startswith("forward_return_")
    ]
    return frame[columns].copy()


def _write_factor_table(path: Path, frame: pd.DataFrame) -> None:
    _validate_factor_table(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def _replace_processed_factor_table(
    *,
    target: Path,
    frame: pd.DataFrame,
    run_id: str,
    backup_existing: bool,
    run_dir: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.stem}.{run_id}.tmp.parquet")
    _write_factor_table(temp_path, frame)
    _validate_factor_table(pd.read_parquet(temp_path, engine="pyarrow"))
    backup_path = None
    if target.exists() and backup_existing:
        backup_path = target.with_name(f"{target.stem}.{run_id}.backup.parquet")
        shutil.copy2(target, backup_path)
    temp_path.replace(target)
    manifest: dict[str, Any] = {
        "status": "ready",
        "overwrite_processed": True,
        "backup_existing": bool(backup_existing),
        "row_count": int(len(frame)),
    }
    if backup_path is not None:
        manifest["backup_file_name"] = backup_path.name
    write_json_sanitized(run_dir / "overwrite_manifest.json", manifest)


def _validate_factor_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("factor_validation_failed")
    missing = [column for column in ("stock_code", "trade_date") if column not in frame]
    if missing:
        raise ValueError("factor_validation_failed")
    if frame.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("factor_validation_failed")
    required_any = set(expected_literature_price_volume_factor_columns()).intersection(
        frame.columns
    )
    if not required_any:
        raise ValueError("invalid_feature_preset")
    label_columns = [
        column
        for column in frame.columns
        if column.startswith("forward_return_")
    ]
    if not label_columns:
        raise ValueError("missing_factor_field")


def _write_feature_set_manifest(
    *,
    run_dir: Path,
    feature_columns: tuple[str, ...],
    unsupported_factors: tuple[dict[str, object], ...],
    warnings: tuple[str, ...],
    row_count: int,
    factor_set_tier: str,
) -> None:
    write_json_sanitized(
        run_dir / "feature_set_manifest.json",
        {
            "feature_set_id": LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
            "factor_pack": LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID,
            "factor_set_tier": factor_set_tier,
            "status": "ready" if feature_columns else "pending",
            "row_count": int(row_count),
            "feature_columns": list(feature_columns),
            "feature_count": len(feature_columns),
            "unsupported_factors": list(unsupported_factors),
            "unsupported_factor_count": len(unsupported_factors),
            "warnings": list(warnings),
            "excludes": [
                "target",
                "label",
                "forward_return",
                "filter",
                "tradability",
                "raw_identifier",
            ],
        },
    )


def _write_factor_eda(
    *,
    run_dir: Path,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> None:
    eda_dir = run_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    if feature_columns:
        missingness = (
            frame[list(feature_columns)]
            .isna()
            .mean()
            .rename("missing_fraction")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
    else:
        missingness = pd.DataFrame(columns=["feature", "missing_fraction"])
    missingness.to_csv(eda_dir / "factor_missingness_by_feature.csv", index=False)
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    coverage = (
        frame.assign(year=dates.dt.year)
        .groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )
    coverage.to_csv(eda_dir / "factor_coverage_by_year.csv", index=False)
    chart_count, chart_status, chart_files = _write_factor_charts(
        eda_dir,
        coverage,
        missingness,
    )
    write_json_sanitized(
        eda_dir / "eda_manifest.json",
        {
            "status": "ready",
            "eda_status": "experimental_non_contract",
            "artifact_count": 3 + chart_count,
            "chart_count": chart_count,
            "chart_dependency_status": chart_status,
            "chart_files": list(chart_files),
        },
    )
    (eda_dir / "eda_report.md").write_text(
        "# Factor rebuild EDA\n\n"
        "Experimental aggregate-only diagnostics; not schema 0.7 contract.\n",
        encoding="utf-8",
    )


def refresh_factor_rebuild_eda(run_dir: Path) -> dict[str, object]:
    """Regenerate aggregate EDA outputs and charts for an existing factor run."""

    if not path_is_under_private_output_root(run_dir, "a_share_factor_rebuilds"):
        raise ValueError("private_output_policy_violation")
    factor_path = run_dir / "factor_rebuild.parquet"
    if not factor_path.is_file():
        raise FileNotFoundError(factor_path)
    eda_dir = run_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = eda_dir / "factor_coverage_by_year.csv"
    missingness_path = eda_dir / "factor_missingness_by_feature.csv"
    dates = pd.to_datetime(
        pd.read_parquet(
            factor_path,
            columns=("trade_date",),
            engine="pyarrow",
        )["trade_date"],
        errors="coerce",
    )
    coverage = (
        pd.DataFrame({"year": dates.dt.year})
        .groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )
    coverage.to_csv(coverage_path, index=False)
    feature_columns = _feature_columns_from_manifest(run_dir)
    if feature_columns:
        frame = pd.read_parquet(
            factor_path,
            columns=feature_columns,
            engine="pyarrow",
        )
        missingness = (
            frame[list(feature_columns)]
            .isna()
            .mean()
            .rename("missing_fraction")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
    else:
        missingness = pd.DataFrame(columns=["feature", "missing_fraction"])
    missingness.to_csv(missingness_path, index=False)
    chart_count, chart_status, chart_files = _write_factor_charts(
        eda_dir,
        coverage,
        missingness,
    )
    write_json_sanitized(
        eda_dir / "eda_manifest.json",
        {
            "status": "ready",
            "eda_status": "experimental_non_contract",
            "artifact_count": 3 + chart_count,
            "chart_count": chart_count,
            "chart_dependency_status": chart_status,
            "chart_files": list(chart_files),
        },
    )
    (eda_dir / "eda_report.md").write_text(
        "# Factor rebuild EDA\n\n"
        "Experimental aggregate-only diagnostics; not schema 0.7 contract.\n",
        encoding="utf-8",
    )
    _write_checkpoints(run_dir)
    return {
        "status": "ready",
        "chart_count": chart_count,
        "chart_dependency_status": chart_status,
        "chart_files": tuple(chart_files),
    }


def _feature_columns_from_manifest(run_dir: Path) -> tuple[str, ...]:
    manifest_path = run_dir / "feature_set_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    columns = manifest.get("feature_columns", [])
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        raise ValueError("invalid_factor_rebuild_config")
    return tuple(columns)


def _write_factor_charts(
    eda_dir: Path,
    coverage: pd.DataFrame,
    missingness: pd.DataFrame,
) -> tuple[int, str, tuple[str, ...]]:
    _clear_factor_chart_files(eda_dir)
    chart_files: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except (ImportError, OSError, RuntimeError):
        _clear_factor_chart_files(eda_dir)
        return 0, "dependency_unavailable", ()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if not coverage.empty:
                fig, ax = plt.subplots(figsize=(6, 3))
                coverage.plot.bar(x="year", y="row_count", ax=ax, legend=False)
                ax.set_xlabel("year")
                ax.set_ylabel("row_count")
                _safe_tight_layout(fig)
                filename = "factor_coverage_by_year.png"
                chart_files.append(filename)
                fig.savefig(eda_dir / filename, dpi=120)
                plt.close(fig)
            if not missingness.empty:
                top = missingness.sort_values(
                    "missing_fraction",
                    ascending=False,
                ).head(20)
                fig, ax = plt.subplots(figsize=(7, 3))
                top.plot.bar(x="feature", y="missing_fraction", ax=ax, legend=False)
                ax.set_xlabel("feature")
                ax.set_ylabel("missing_fraction")
                _safe_tight_layout(fig)
                filename = "factor_missingness_top20.png"
                chart_files.append(filename)
                fig.savefig(eda_dir / filename, dpi=120)
                plt.close(fig)
    except (ImportError, OSError, RuntimeError):
        _clear_factor_chart_files(eda_dir)
        return 0, "dependency_unavailable", ()
    return (
        len(chart_files),
        "ready" if chart_files else "not_generated",
        tuple(chart_files),
    )


def _clear_factor_chart_files(eda_dir: Path) -> None:
    for filename in (
        "factor_coverage_by_year.png",
        "factor_missingness_top20.png",
    ):
        (eda_dir / filename).unlink(missing_ok=True)


def _safe_tight_layout(fig: object) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()


def _runtime_summary(
    run_id: str,
    status: str,
    failure_code: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "run_id": run_id,
        "factor_pack": LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID,
        "boundary": PREDICTION_SAFE_BOUNDARY,
    }
    if failure_code:
        payload["failure_code"] = failure_code
    return payload


def _degraded_run_warnings(
    config: FactorRebuildConfig,
    warnings: tuple[str, ...],
    warmup: dict[str, object],
) -> tuple[str, ...]:
    messages = list(warnings)
    calculation_start = _calculation_start_date(config)
    evaluation_start = _evaluation_start_date(config)
    if (
        calculation_start
        and evaluation_start
        and pd.Timestamp(calculation_start) < pd.Timestamp(evaluation_start)
    ):
        messages.append(str(warmup["warmup_status"]))
    elif warmup["warmup_status"] == "date_filtered_without_warmup":
        messages.append("date_filtered_without_warmup")
    return tuple(messages)


def _warmup_status(config: FactorRebuildConfig) -> str:
    calculation_start = _calculation_start_date(config)
    evaluation_start = _evaluation_start_date(config)
    if (
        calculation_start
        and evaluation_start
        and pd.Timestamp(calculation_start) < pd.Timestamp(evaluation_start)
    ):
        return "warmup_aware_unchecked"
    if config.start_date or config.end_date:
        return "date_filtered_without_warmup"
    return "full_range"


def _write_failed_run_summary(run_dir: Path, run_id: str, failure_code: str) -> None:
    write_json_sanitized(
        run_dir / "runtime_summary.json",
        _runtime_summary(run_id, "failed", failure_code),
    )


def _failure_code(exc: BaseException) -> str:
    detail = str(exc)
    known = {
        "memory_preflight_failed",
        "factor_validation_failed",
        "invalid_feature_preset",
        "missing_factor_field",
        "duplicate_run_id",
        "private_output_policy_violation",
        "invalid_factor_rebuild_config",
        "warmup_insufficient",
    }
    if detail in known:
        return detail
    if isinstance(exc, FileNotFoundError):
        return "missing_factor_input"
    return "factor_rebuild_failed"


def _write_checkpoints(run_dir: Path) -> None:
    checkpoints = [
        ("schema_inventory", "schema_inventory.json"),
        ("memory_preflight", "memory_preflight.json"),
        ("warmup_sufficiency", "warmup_sufficiency.json"),
        ("factor_rebuild", "factor_rebuild.parquet"),
        ("feature_set_manifest", "feature_set_manifest.json"),
        ("overwrite_manifest", "overwrite_manifest.json"),
        ("eda", "eda_manifest"),
        ("runtime_summary", "runtime_summary.json"),
    ]
    payload = [
        {
            "key": key,
            "label": path,
            "status": "ready" if _checkpoint_ready(run_dir, key) else "pending",
        }
        for key, path in checkpoints
    ]
    write_json_sanitized(run_dir / "checkpoints.json", {"checkpoints": payload})


def _checkpoint_ready(run_dir: Path, key: str) -> bool:
    paths = {
        "schema_inventory": run_dir / "schema_inventory.json",
        "memory_preflight": run_dir / "memory_preflight.json",
        "warmup_sufficiency": run_dir / "warmup_sufficiency.json",
        "factor_rebuild": run_dir / "factor_rebuild.parquet",
        "feature_set_manifest": run_dir / "feature_set_manifest.json",
        "overwrite_manifest": run_dir / "overwrite_manifest.json",
        "eda": run_dir / "eda" / "eda_manifest.json",
        "runtime_summary": run_dir / "runtime_summary.json",
    }
    return paths[key].is_file()


__all__ = [
    "FACTOR_REBUILD_INPUT_COLUMNS",
    "FactorRebuildConfig",
    "FactorRebuildResult",
    "build_warmup_sufficiency_payload",
    "refresh_factor_rebuild_eda",
    "run_factor_rebuild",
    "write_schema_inventory_run",
]
